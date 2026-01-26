#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Run compiled .so files from tuning results (OpenCL/Vulkan/CPU)
==============================================================

This script loads and runs compiled shared objects (.so) generated from
OpenCL tuning. It can run the models on a remote device via RPC or locally.

Usage:
    # Run a specific .so file on remote device
    python run_so.py --so path/to/matmul_MxKxN_candidate.so

    # Run all .so files in a directory
    python run_so.py --dir ./tuning_logs/20251115_184133/

    # Run with custom matrix dimensions
    python run_so.py --so model.so --m 256 --k 1024 --n 512

    # Run on CPU backend
    python run_so.py --so model.so --backend cpu

Example:
    python run_so.py --so tuning_logs/20251115_184133/matmul_257x1024x3072_cand001-base.so
"""

import argparse
import csv
import datetime
import glob
import os
import re
import time
import subprocess
from contextlib import contextmanager

import numpy as np
import tvm
from tvm import rpc
from tvm.contrib import ndk


def parse_shape_from_filename(filename):
    """Extract M, K, N dimensions from filename pattern like matmul_MxKxN_*.so
    
    Parameters
    ----------
    filename : str
        Filename containing shape information
        
    Returns
    -------
    tuple or None
        (M, K, N) if found, None otherwise
    """
    match = re.search(r'(\d+)x(\d+)x(\d+)', filename)
    if match:
        m, k, n = match.groups()
        return int(m), int(k), int(n)
    return None

def _adb_cmd(serial: str | None = None):
    return ["adb"] if not serial else ["adb", "-s", serial]


def adb_write_label(text: str, serial: str | None = None) -> None:
    """Write label text to /data/local/tmp/label on the Android device via adb.

    Best-effort; raises on failure, call site should handle.
    """
    cmd = _adb_cmd(serial) + ["shell", "sh", "-c", "cat > /data/local/tmp/label"]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True, timeout=10)


def adb_remove_label(serial: str | None = None) -> None:
    """Remove the label file from the Android device via adb (best-effort)."""
    subprocess.run(_adb_cmd(serial) + ["shell", "rm", "-f", "/data/local/tmp/label"], check=True, timeout=10)


@contextmanager
def android_label(text: str, serial: str | None = None, enable: bool = False):
    """Context manager that writes an adb label before work and removes it afterwards.

    Always attempts cleanup even if writing failed or the body raises.
    """
    if not enable:
        yield
        return
    wrote = False
    try:
        try:
            adb_write_label(text, serial)
            wrote = True
            print("✓ ADB label written to /data/local/tmp/label")
        except Exception as e:
            print(f"WARN: Failed to write ADB label: {e}")
        yield
    finally:
        try:
            adb_remove_label(serial)
            if wrote:
                print("✓ ADB label removed")
        except Exception as e:
            print(f"WARN: Failed to remove ADB label: {e}")


def run_so(so_path, m, k, n, tracker_host, tracker_port, tracker_key, 
                  backend="opencl", verify=True, number=10, repeat=20, warmup_runs=5,
                  adb_label=False, adb_serial: str | None = None):
    """Run a compiled .so file on remote device via RPC
    
    Parameters
    ----------
    so_path : str
        Path to the .so file
    m, k, n : int
        Matrix dimensions for A[M,K] @ B[K,N]
    tracker_host : str
        RPC tracker host address
    tracker_port : int
        RPC tracker port
    tracker_key : str
        RPC device key
    backend : str
        Device backend ("opencl" or "vulkan")
    verify : bool
        Whether to verify result correctness
    number : int
        Number of times to run for each measurement
    repeat : int
        Number of measurements to take
        
    Returns
    -------
    dict
        Results containing timing and success status
    """
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"Running remotely: {os.path.basename(so_path)}")
    print(f"Shape: {m}x{k}x{n}, Backend: {backend}")
    print(f"RPC: {tracker_host}:{tracker_port}, key={tracker_key}")
    print(separator)
    
    remote = None
    remote_filename = None
    

    try:
        label_lines = [
            os.path.basename(so_path),
        ]
        label_text = "\n".join(label_lines) + "\n"

        # Setup and warmup (no label)
        # Connect to RPC
        tracker = rpc.connect_tracker(tracker_host, tracker_port)
        remote = tracker.request(tracker_key, priority=1, session_timeout=300)
        print(f"✓ Connected to remote device")

        # Upload and load module
        remote.upload(so_path)
        remote_filename = os.path.basename(so_path)
        mod = remote.load_module(remote_filename)
        print(f"✓ Module loaded on remote device")

        # Get device context
        if backend == "vulkan":
            try:
                dev = remote.vulkan(0)
            except Exception as e:
                print(f"WARN: Vulkan not available, falling back to OpenCL: {e}")
                dev = remote.cl(0)
        elif backend == "opencl":
            dev = remote.cl(0)
        else:  # cpu
            config_func = remote.get_function('runtime.config_threadpool')

            # use single thread, BIG core only
            # mode: 0 (all) / 1 (big only) / -1 (little only)
            # nthreads: 0 (all available) / N (n threads)
            mode = 1
            nthreads = 1
            config_func(mode, nthreads)

            dev = remote.cpu(0)

        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create input tensors
        a_np = np.random.uniform(size=(m, k)).astype(np.float32)
        b_np = np.random.uniform(size=(k, n)).astype(np.float32)
        c_np = np.zeros((m, n), dtype=np.float32)

        a = tvm.runtime.tensor(a_np, dev)
        b = tvm.runtime.tensor(b_np, dev)
        c = tvm.runtime.tensor(c_np, dev)
        print(f"✓ Tensors allocated on device")

        # Get function (try common names)
        func_names = ["matmul", "main", "entry_name", "default_function"]
        func = None
        func_name = None
        for name in func_names:
            try:
                func = mod[name]
                print(f"✓ Found function: {name}")
                func_name = name
                break
            except:
                continue

        if func is None:
            print(f"ERROR: Could not find entry function")
            return {
                "success": False,
                "error": "Function not found"
            }

        # Warm up (no label)
        for _ in range(max(0, int(warmup_runs))):
            func(a, b, c)

        # Verify correctness (no label)
        if verify:
            c_result = c.numpy()
            c_expected = np.dot(a_np, b_np)
            try:
                np.testing.assert_allclose(c_result, c_expected, rtol=1e-4, atol=1e-4)
                print("✓ Correctness check passed")
            except AssertionError as e:
                print(f"✗ Correctness check failed: {e}")
                return {
                    "success": False,
                    "error": "Correctness check failed"
                }

        # Benchmark - only here do we write the label
        if func_name is None:
            func_name = func_names[0]  # fallback

        with android_label(label_text, serial=adb_serial, enable=adb_label):
            time_f = mod.time_evaluator(func_name, dev, number=number, repeat=repeat)
            timing_result = time_f(a, b, c)

            times_s = timing_result.results
            mean_ms = float(np.mean(times_s)) * 1000.0
            std_ms = float(np.std(times_s)) * 1000.0
            min_ms = float(np.min(times_s)) * 1000.0
            max_ms = float(np.max(times_s)) * 1000.0

            print(f"Mean: {mean_ms:.3f} ± {std_ms:.3f} ms")
            print(f"Min:  {min_ms:.3f} ms")
            print(f"Max:  {max_ms:.3f} ms")

            return {
                "success": True,
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "min_ms": min_ms,
                "max_ms": max_ms
            }

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }
        
    finally:
        # Cleanup
        if remote and remote_filename:
            try:
                remote.remove(remote_filename)
                print("✓ Remote file cleaned up")
            except:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Run compiled .so files from tuning results (auto-detect backend from filename)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options
    parser.add_argument("--so", type=str, help="Path to a specific .so file")
    parser.add_argument("--dir", type=str, help="Directory containing .so files")
    
    # Matrix dimensions (optional, auto-detected from filename if not provided)
    parser.add_argument("--m", type=int, help="Matrix dimension M")
    parser.add_argument("--k", type=int, help="Matrix dimension K")
    parser.add_argument("--n", type=int, help="Matrix dimension N")
    
    # RPC configuration
    parser.add_argument("--tracker-host", type=str, default="127.0.0.1",
                        help="RPC tracker host (default: 127.0.0.1)")
    parser.add_argument("--tracker-port", type=int, default=9190,
                        help="RPC tracker port (default: 9190)")
    parser.add_argument("--tracker-key", type=str, default="android64",
                        help="RPC device key (default: android64)")
    
    # Backend
    parser.add_argument("--backend", type=str, default=None,
                        choices=["opencl", "vulkan", "cpu"],
                        help="Device backend. If omitted, auto-detect from filename (_opencl_, _vulkan_, _cpu_). Default: auto-detect, fallback opencl")
    
    # Benchmark options
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip correctness verification")
    parser.add_argument("--number", type=int, default=50,
                        help="Number of times to run for each measurement (default: 50)")
    parser.add_argument("--repeat", type=int, default=50,
                        help="Number of measurements to take (default: 50)")
    parser.add_argument("--warmup-runs", type=int, default=10,
                        help="Number of warm-up runs before verification and timing (default: 10)")

    # ADB label options
    parser.add_argument("--adb-label", action="store_true",
                        help="Write kernel info to /data/local/tmp/label on Android via adb before running, and remove it after.")
    parser.add_argument("--adb-serial", type=str, default=None,
                        help="ADB device serial (optional). If not set, uses default adb device.")
    
    # Output
    parser.add_argument("--csv", type=str, help="Save results to CSV file")
    parser.add_argument("--ref-csv", type=str, default=None,
                        help="Path to a reference CSV to verify times against (if omitted and --dir is used, will auto-detect bench_*.csv in the dir)")
    parser.add_argument("--time-rtol", type=float, default=0.2,
                        help="Relative tolerance for time verification (default: 0.2)")
    parser.add_argument("--time-atol", type=float, default=0.005,
                        help="Absolute tolerance in milliseconds for time verification (default: 0.005 ms)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.so and not args.dir:
        parser.error("Either --so or --dir must be specified")
    
    # Collect .so files
    so_files = []
    if args.so:
        if not os.path.exists(args.so):
            print(f"ERROR: File not found: {args.so}")
            return 1
        so_files.append(args.so)
    else:
        pattern = os.path.join(args.dir, "*.so")
        so_files = sorted(glob.glob(pattern))
        if not so_files:
            print(f"ERROR: No .so files found in: {args.dir}")
            return 1
        print(f"Found {len(so_files)} .so files")
    
    # Results collection
    results = []
    csv_header = ["filename", "m", "k", "n", "backend",
                  "mean_ms", "std_ms", "min_ms", "max_ms", "success", "error"]

    def parse_backend_from_filename(path: str):
        base = os.path.basename(path).lower()
        if "_opencl_" in base or base.endswith("_opencl.so") or base.endswith("_opencl.tar"):
            return "opencl"
        if "_vulkan_" in base or base.endswith("_vulkan.so") or base.endswith("_vulkan.tar"):
            return "vulkan"
        if "_cpu_" in base or base.endswith("_cpu.so") or base.endswith("_cpu.tar"):
            return "cpu"
        return None

    def parse_candidate_from_filename(path: str):
        base = os.path.basename(path).lower()
        m = re.search(r"(cand\d+)", base)
        return m.group(1) if m else None

    # Load reference CSV if provided or auto-detect when running a directory
    ref_rows = []
    if args.ref_csv and os.path.exists(args.ref_csv):
        try:
            with open(args.ref_csv, "r") as f:
                reader = csv.DictReader(f)
                ref_rows = list(reader)
        except Exception as e:
            print(f"WARN: Failed to read reference CSV {args.ref_csv}: {e}")
    elif args.dir:
        # Auto-detect bench CSVs inside the directory
        for p in glob.glob(os.path.join(args.dir, "bench_*.csv")):
            try:
                with open(p, "r") as f:
                    reader = csv.DictReader(f)
                    ref_rows.extend(list(reader))
            except Exception:
                pass

    # Build lookup maps from reference rows
    ref_by_filename = {}
    ref_by_key = {}  # (candidate, backend, M, K, N, source)
    if ref_rows:
        for r in ref_rows:
            # Normalize keys
            lower = {k.strip(): v for k, v in r.items()}
            # filename-based
            fname = lower.get("filename") or lower.get("file")
            if fname:
                ref_by_filename[os.path.basename(fname)] = lower
            # candidate-based
            try:
                cand = lower.get("candidate")
                backend_ref = (lower.get("backend") or "").lower()
                source = (lower.get("source") or "").lower()
                M = int(float(lower.get("m") or lower.get("M") or 0))
                K = int(float(lower.get("k") or lower.get("K") or 0))
                N = int(float(lower.get("n") or lower.get("N") or 0))
                if cand and backend_ref and M and K and N:
                    key = (cand.lower(), backend_ref, M, K, N, source)
                    ref_by_key[key] = lower
            except Exception:
                continue

    def find_ref_row(so_path: str, m: int, k: int, n: int, backend: str):
        # 1) by filename
        base = os.path.basename(so_path)
        if base in ref_by_filename:
            return ref_by_filename[base]
        # 2) by candidate + shape + backend; prefer run_so then internal
        cand = parse_candidate_from_filename(so_path)
        if cand:
            for source in ("run_so", "internal", ""):
                key = (cand.lower(), backend.lower(), m, k, n, source)
                if key in ref_by_key:
                    return ref_by_key[key]
        return None

    # Run each .so file
    for so_path in so_files:
        if "neon+dotprod" not in os.path.basename(so_path).lower():
            continue
# capture candidate number (candXX)
        # Determine dimensions
        if args.m and args.k and args.n:
            m, k, n = args.m, args.k, args.n
        else:
            shape = parse_shape_from_filename(so_path)
            if shape:
                m, k, n = shape
            else:
                print(f"WARN: Could not parse shape from filename: {so_path}")
                print(f"      Please provide --m, --k, --n arguments")
                continue

        # # filter shape here
        # targets = [
        #     (1, 1024, 3072),
        #     (1, 1536, 8960),
        # ]

        # if (m, k, n) not in targets:
        #     print(f"Skipping shape ({m},{k},{n}) not in target list.")
        #     continue

        

        # mnk specific configs (sleep duration, number and repeat adjustments)
        mkn_number_repeat_table = {
            (1, 1024, 3072): (80, 80),
            (1, 1024, 4096): (100, 100),
            (1, 1536, 2048): (150, 100),
            (1, 1536, 8960): (50, 50),
            (1, 8192, 8192): (20, 20),
        }

        assert (m, k, n) in mkn_number_repeat_table.keys(), \
            f"WARN: mkn ({m},{k},{n}) not in the predefined table. Please update the table accordingly."
        args.number, args.repeat = mkn_number_repeat_table[(m, k, n)]
        print(f"Using number={args.number}, repeat={args.repeat} for shape ({m},{k},{n})")

        # check the device temperature, and busy wait until it cools down
        # temperature checking command: `adb shell cat /sys/class/thermal/thermal_zone53/temp`
        # expected output format: 23791 (/1000 to get °C)
        # parse the first float number after 'SKIN: ', and keep the value below 30.0
        temp_cmd = _adb_cmd(args.adb_serial) + ["shell", "cat", "/sys/class/thermal/thermal_zone53/temp"]
        sleep_start_temperature = 30.0  # Celsius
        sleep_end_temperature = 30.0  # Celsius

        sleep_on = False
        while True:
            try:
                output = subprocess.check_output(temp_cmd, timeout=10).decode("utf-8")
                temp_milli = int(re.search(r'(\d+)', output).group(1))
                temp_c = temp_milli / 1000.0
                print(f"Current device temperature: {temp_c:.2f} °C")

                if sleep_on:
                    if temp_c <= sleep_end_temperature:
                        print(f"✓ Device cooled down to {temp_c:.2f} °C, resuming execution.")
                        break
                    else:
                        print(f"Waiting for device to cool down below {sleep_end_temperature} °C...")
                else:
                    if temp_c >= sleep_start_temperature:
                        sleep_on = True
                        print(f"Device temperature {temp_c:.2f} °C exceeds {sleep_start_temperature} °C, waiting to cool down...")
                    else:
                        break
            except Exception as e:
                print(f"WARN: Failed to get device temperature: {e}")
                break
            time.sleep(10)  # wait before re-checking
        print("Starting execution...") 
        
        # Decide backend for this file
        effective_backend = args.backend or parse_backend_from_filename(so_path) or "opencl"
        if args.backend is None and parse_backend_from_filename(so_path) is None:
            print(f"WARN: Could not detect backend from filename; defaulting to opencl. Consider --backend option.")

        # Run
        result = run_so(
            so_path, m, k, n,
            tracker_host=args.tracker_host,
            tracker_port=args.tracker_port,
            tracker_key=args.tracker_key,
            backend=effective_backend,
            verify=not args.no_verify,
            number=args.number,
            repeat=args.repeat,
            warmup_runs=args.warmup_runs,
            adb_label=args.adb_label,
            adb_serial=args.adb_serial
        )

        # Optional: verify timing vs reference CSV
        ref_mean = None
        verify_pass = None
        verify_delta = None
        ref_row = find_ref_row(so_path, m, k, n, effective_backend) if ref_rows else None
        if ref_row is not None:
            # Use column name mean_ms (case-insensitive), fallback to mean
            mean_str = ref_row.get("mean_ms") or ref_row.get("MEAN_MS") or ref_row.get("mean")
            try:
                ref_mean = float(mean_str)
            except Exception:
                ref_mean = None
            if ref_mean is not None and result.get("mean_ms") is not None:
                delta = abs(float(result["mean_ms"]) - ref_mean)
                verify_delta = delta
                tol = args.time_atol + args.time_rtol * abs(ref_mean)
                verify_pass = delta <= tol
                status = "PASS" if verify_pass else "FAIL"
                print(f"Verify vs CSV: ref={ref_mean:.6f} ms, measured={result['mean_ms']:.6f} ms, "
                      f"delta={delta:.6f} ms, tol={tol:.6f} ms -> {status}")
        
        # Collect result
        row = [
            os.path.basename(so_path),
            m, k, n,
            effective_backend,
            result.get("mean_ms"),
            result.get("std_ms"),
            result.get("min_ms"),
            result.get("max_ms"),
            result.get("success"),
            result.get("error", "")
        ]
        # Append verification columns if a reference is available
        if ref_rows:
            if "ref_mean_ms" not in csv_header:
                csv_header.extend(["ref_mean_ms", "verify_pass", "verify_delta_ms"])
            row.extend([ref_mean, verify_pass, verify_delta])
        results.append(row)
        
        # Cool down between runs
        if len(so_files) > 1:
            time.sleep(2)
    
    # Print summary
    separator = "=" * 60
    print(f"\n{separator}")
    print("SUMMARY")
    print(separator)
    
    success_count = sum(1 for r in results if r[-2])  # success column
    print(f"Total: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    if ref_rows:
        v_cols = None
        verify_passes = 0
        verify_total = 0
        # Determine indices for appended columns if present
        if "ref_mean_ms" in csv_header and "verify_pass" in csv_header and "verify_delta_ms" in csv_header:
            i_ref = csv_header.index("ref_mean_ms")
            i_vp = csv_header.index("verify_pass")
            i_vd = csv_header.index("verify_delta_ms")
            for r in results:
                if len(r) > i_vp and r[i_ref] is not None:
                    verify_total += 1
                    if r[i_vp]:
                        verify_passes += 1
        print(f"Verify: {verify_passes}/{verify_total} within tolerance (rtol={args.time_rtol}, atol={args.time_atol} ms)")
    
    # Show successful runs sorted by mean time
    successful = [r for r in results if r[-2]]
    if successful:
        print(f"\nSuccessful runs (sorted by mean time):")
        successful.sort(key=lambda x: x[5] if x[5] is not None else float('inf'))
        for r in successful:
            print(f"  {r[0]}: {r[5]:.3f} ± {r[6]:.3f} ms")
    
    # Save to CSV if requested
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            for row in results:
                writer.writerow(row)
        print(f"\n✓ Results saved to: {args.csv}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
