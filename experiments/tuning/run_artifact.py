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
Load and run kernel artifacts from tuning logs.

This script provides a unified interface to load and run kernel artifacts:
1. Compiled kernel artifacts (.tar and .so files) - directly loadable TVM modules
   - .tar files are automatically converted to .so files locally before uploading
   - Requires TVM_NDK_CC environment variable to be set for .tar conversion
2. Trace files (trace_*.py) - schedule traces that need to be applied to a workload

This script reuses functionality from run_py.py and run_so.py.

Note: For .tar files, TVM_NDK_CC must be set to convert them to .so files
      before uploading to the Android device (which doesn't have g++).

Usage:
    # Run a specific .tar artifact (requires TVM_NDK_CC)
    export TVM_NDK_CC=/path/to/android-ndk/.../aarch64-linux-android21-clang++
    python run_artifact.py --artifact tuning_logs/opencl_pixel8pro/1x1536x8960_cand1000_base.tar
    
    # Run a trace file
    python run_artifact.py --artifact tuning_logs/opencl_pixel8pro/trace_197x768x2304_cand001.py
    
    # Run all artifacts in a directory
    python run_artifact.py --dir tuning_logs/opencl_pixel8pro/
    
    # Run with custom RPC settings
    python run_artifact.py --artifact artifact.tar --tracker-host 127.0.0.1 --tracker-port 9190 --tracker-key android64
"""

import argparse
import csv
import datetime
import glob
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from typing import Optional, Tuple

import numpy as np
import tvm
from tvm import rpc, te, tir
from tvm.contrib import ndk

# Import from run_so.py
from run_so import run_so, parse_shape_from_filename

# Import from run_py.py
from run_py import get_matmul_workload, load_schedule_from_trace_file

# Import bandwidth profiler
try:
    from gpu_bandwidth_profiler import GPUBandwidthClient
except ImportError:
    GPUBandwidthClient = None
    print("Warning: gpu_bandwidth_profiler not available. Bandwidth measurement will be disabled.")


def parse_backend_from_filename(path: str) -> Optional[str]:
    """Parse backend from filename or directory name.
    
    Parameters
    ----------
    path : str
        Path to artifact file
        
    Returns
    -------
    str or None
        Backend name ("opencl", "vulkan", "cpu") or None if not found
    """
    base = os.path.basename(path).lower()
    if "_opencl_" in base or base.endswith("_opencl.so") or base.endswith("_opencl.tar"):
        return "opencl"
    if "_vulkan_" in base or base.endswith("_vulkan.so") or base.endswith("_vulkan.tar"):
        return "vulkan"
    if "_cpu_" in base or base.endswith("_cpu.so") or base.endswith("_cpu.tar"):
        return "cpu"
    
    # Try to parse from directory name
    dir_name = os.path.dirname(path).lower()
    if "opencl" in dir_name:
        return "opencl"
    if "vulkan" in dir_name:
        return "vulkan"
    if "cpu" in dir_name:
        return "cpu"
    
    return None


def convert_tar_to_so(tar_path: str, backend: str = "opencl", timeout_event: Optional[threading.Event] = None) -> str:
    """Convert a .tar artifact to .so by extracting and linking object files.
    
    This function extracts object files from a .tar archive and links them
    using the Android NDK compiler (TVM_NDK_CC). This is necessary because
    .tar files contain object files that need to be linked, but Android devices
    don't have g++ available for linking on the device.
    
    Parameters
    ----------
    tar_path : str
        Path to .tar file
    backend : str
        Backend name (for error messages)
    timeout_event : threading.Event, optional
        Event to check for timeout
        
    Returns
    -------
    str
        Path to generated .so file
        
    Raises
    ------
    RuntimeError
        If conversion fails
    """
    if timeout_event and timeout_event.is_set():
        raise RuntimeError("Operation timed out before tar conversion")
    
    ndk_cc = os.environ.get("TVM_NDK_CC")
    if not ndk_cc:
        raise RuntimeError(
            "TVM_NDK_CC not set. Cannot convert .tar to .so without NDK compiler.\n"
            "Set TVM_NDK_CC to your Android NDK clang++ path.\n"
            "Example: export TVM_NDK_CC=/path/to/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++"
        )
    
    if not os.path.exists(ndk_cc):
        raise RuntimeError(f"TVM_NDK_CC path does not exist: {ndk_cc}")
    
    print(f"Converting .tar to .so: {os.path.basename(tar_path)}")
    
    # Extract tar file
    tmp_extract_dir = tempfile.mkdtemp()
    try:
        if timeout_event and timeout_event.is_set():
            raise RuntimeError("Operation timed out during tar extraction")
        
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(tmp_extract_dir)
    except Exception as e:
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to extract tar file {tar_path}: {e}")
    
    # Find object files
    object_files = glob.glob(os.path.join(tmp_extract_dir, "*.o"))
    if not object_files:
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)
        raise RuntimeError(f"No object files found in extracted tar: {tmp_extract_dir}")
    
    if timeout_event and timeout_event.is_set():
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)
        raise RuntimeError("Operation timed out after tar extraction")
    
    # Link object files into .so
    base_name = os.path.splitext(os.path.basename(tar_path))[0]
    so_dir = tempfile.mkdtemp()
    so_path = os.path.join(so_dir, f"{base_name}.so")
    
    compile_cmd = [ndk_cc, "-shared", "-fPIC", "-o", so_path] + object_files
    
    try:
        if timeout_event and timeout_event.is_set():
            raise RuntimeError("Operation timed out before compilation")
        
        result = subprocess.run(
            compile_cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for compilation
        )
        print(f"✓ Converted to: {so_path}")
        return so_path
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)
        shutil.rmtree(so_dir, ignore_errors=True)
        raise RuntimeError(f"Tar to SO conversion timed out after 300 seconds")
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)
        shutil.rmtree(so_dir, ignore_errors=True)
        raise RuntimeError(
            f"Tar to SO conversion failed: Compilation error:\n{e.stdout}{e.stderr}\n"
            f"Command line: {' '.join(compile_cmd)}"
        )
    finally:
        # Clean up extraction directory (but keep .so file)
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)


def build_from_trace(trace_file: str, shape: Tuple[int, int, int], backend: str = "opencl") -> str:
    """Build a module from a trace file.
    
    Parameters
    ----------
    trace_file : str
        Path to trace_*.py file
    shape : tuple
        (M, K, N) matrix dimensions
    backend : str
        Backend name ("opencl", "vulkan", "cpu")
        
    Returns
    -------
    str
        Path to generated .so file
    """
    m, k, n = shape
    
    # Create workload
    A, B, C = get_matmul_workload(m, k, n)
    func = te.create_prim_func([A, B, C]).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    
    # Load schedule from trace
    sch = load_schedule_from_trace_file(trace_file, mod, func_name="main")
    
    # Build target
    host_triple = "llvm -mtriple=aarch64-linux-android"
    if backend == "cpu" or backend == "llvm":
        target = tvm.target.Target(host_triple)
    elif backend == "vulkan":
        target = tvm.target.Target("vulkan", host=host_triple)
    else:  # opencl
        target = tvm.target.Target("opencl -device=adreno", host=host_triple)
    
    # Build module
    rt_mod = tvm.build(sch.mod, target=target)
    
    # Export to .so
    out_dir = tempfile.mkdtemp()
    base_name = os.path.splitext(os.path.basename(trace_file))[0]
    out_path = os.path.join(out_dir, f"{base_name}.so")
    
    if backend in ["opencl", "vulkan"] and os.environ.get("TVM_NDK_CC"):
        rt_mod.export_library(out_path, fcompile=ndk.create_shared)
    else:
        rt_mod.export_library(out_path)
    
    return out_path


def run_artifact(
    artifact_path: str,
    shape: Optional[Tuple[int, int, int]] = None,
    tracker_host: str = "127.0.0.1",
    tracker_port: int = 9190,
    tracker_key: str = "android64",
    backend: Optional[str] = None,
    verify: bool = True,
    number: int = 10,
    repeat: int = 20,
    warmup_runs: int = 5,
    timeout_sec: int = 600,
    bandwidth_host: Optional[str] = None,
    bandwidth_port: int = 8888
) -> dict:
    """Run a kernel artifact (tar file, so file, or trace file).
    
    This function handles:
    - .tar and .so files: Uses run_so() from run_so.py
    - trace_*.py files: Builds from trace then uses run_so()
    
    Parameters
    ----------
    artifact_path : str
        Path to artifact (.tar, .so, or trace_*.py file)
    shape : tuple, optional
        (M, K, N) dimensions. Auto-detected from filename if not provided.
    tracker_host : str
        RPC tracker host
    tracker_port : int
        RPC tracker port
    tracker_key : str
        RPC device key
    backend : str, optional
        Backend ("opencl", "vulkan", "cpu"). Auto-detected if not provided.
    verify : bool
        Whether to verify correctness
    number : int
        Number of runs per measurement
    repeat : int
        Number of measurements
    warmup_runs : int
        Number of warmup runs
    timeout_sec : int
        Overall timeout in seconds for the entire operation (default: 600)
    bandwidth_host : str, optional
        GPU bandwidth profiler host (enables bandwidth measurement for GPU backends)
    bandwidth_port : int
        GPU bandwidth profiler port (default: 8888)
        
    Returns
    -------
    dict
        Results with timing and success status, plus bandwidth statistics if enabled
    """
    # Set up timeout handler
    timeout_occurred = threading.Event()
    
    def timeout_handler():
        timeout_occurred.set()
    
    timer = threading.Timer(timeout_sec, timeout_handler)
    timer.start()
    
    try:
        return _run_artifact_impl(
            artifact_path, shape, tracker_host, tracker_port, tracker_key,
            backend, verify, number, repeat, warmup_runs, timeout_occurred,
            bandwidth_host=bandwidth_host, bandwidth_port=bandwidth_port
        )
    finally:
        timer.cancel()


def _run_artifact_impl(
    artifact_path: str,
    shape: Optional[Tuple[int, int, int]],
    tracker_host: str,
    tracker_port: int,
    tracker_key: str,
    backend: Optional[str],
    verify: bool,
    number: int,
    repeat: int,
    warmup_runs: int,
    timeout_occurred: threading.Event,
    bandwidth_host: Optional[str] = None,
    bandwidth_port: int = 8888
) -> dict:
    
    # Determine artifact type
    is_trace = artifact_path.endswith('.py') and 'trace_' in os.path.basename(artifact_path)
    is_tar = artifact_path.endswith('.tar')
    is_so = artifact_path.endswith('.so')
    
    if not (is_trace or is_tar or is_so):
        return {
            "success": False,
            "error": f"Unknown artifact type: {artifact_path}"
        }
    
    # Parse shape from filename if not provided
    if shape is None:
        shape = parse_shape_from_filename(artifact_path)
        if shape is None:
            return {
                "success": False,
                "error": "Could not parse shape from filename. Please provide --m, --k, --n"
            }
    
    m, k, n = shape
    
    # Determine backend
    if backend is None:
        backend = parse_backend_from_filename(artifact_path) or "opencl"
    
    # Map backend names: "llvm" -> "cpu" for run_so compatibility
    backend_for_run_so = "cpu" if backend == "llvm" else backend
    
    # Check timeout before starting operations
    if timeout_occurred.is_set():
        return {
            "success": False,
            "error": "Operation timed out before starting"
        }
    
    # For trace files, build first
    if is_trace:
        try:
            artifact_path = build_from_trace(artifact_path, shape, backend)
            is_tar = False
            is_so = True
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to build from trace: {e}"
            }
    
    # For .tar files, convert to .so
    if is_tar:
        try:
            artifact_path = convert_tar_to_so(artifact_path, backend, timeout_occurred)
            is_so = True
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # Initialize bandwidth profiler if enabled (only for GPU backends)
    bandwidth_client = None
    bandwidth_connected = False
    
    if bandwidth_host and backend_for_run_so in ["opencl", "vulkan"]:
        if GPUBandwidthClient is None:
            print("⚠ Bandwidth profiler not available (gpu_bandwidth_profiler not installed)")
        else:
            try:
                bandwidth_client = GPUBandwidthClient(host=bandwidth_host, port=bandwidth_port, timeout=2.0)
                if bandwidth_client.connect():
                    print(f"✓ Connected to bandwidth profiler at {bandwidth_host}:{bandwidth_port}")
                    bandwidth_client.clear_buffer()
                    bandwidth_connected = True
                else:
                    print(f"⚠ Could not connect to bandwidth profiler, continuing without bandwidth measurement")
                    bandwidth_client = None
            except Exception as e:
                print(f"⚠ Failed to initialize bandwidth profiler: {e}, continuing without bandwidth measurement")
                bandwidth_client = None
    elif backend_for_run_so in ["opencl", "vulkan"]:
        # For GPU backends, try to auto-detect bandwidth profiler on localhost
        if GPUBandwidthClient is not None:
            try:
                bandwidth_client = GPUBandwidthClient(host="127.0.0.1", port=bandwidth_port, timeout=2.0)
                if bandwidth_client.connect():
                    print(f"✓ Auto-connected to bandwidth profiler at 127.0.0.1:{bandwidth_port}")
                    bandwidth_client.clear_buffer()
                    bandwidth_connected = True
                else:
                    bandwidth_client = None
            except Exception:
                # Silently fail if auto-detection doesn't work
                bandwidth_client = None
    
    # Use run_so() for .so files (tar files are now converted to .so)
    # Note: run_so has its own session_timeout, but we add an extra safety check
    max_bandwidth_retries = 2  # Maximum retries if bandwidth is zero
    current_repeat = repeat
    result = None
    
    try:
        for retry_attempt in range(max_bandwidth_retries + 1):
            try:
                # Start bandwidth profiling if connected
                if bandwidth_connected and bandwidth_client:
                    bandwidth_client.clear_buffer()  # Clear any previous data
                    bandwidth_client.start(sampling_interval_ms=10)  # 10ms sampling interval
                    if retry_attempt > 0:
                        print(f"✓ Restarted bandwidth profiling (retry {retry_attempt}, repeat={current_repeat})")
                    else:
                        print("✓ Started bandwidth profiling")
                
                result = run_so(
                    artifact_path,
                    m, k, n,
                    tracker_host=tracker_host,
                    tracker_port=tracker_port,
                    tracker_key=tracker_key,
                    backend=backend_for_run_so,
                    verify=verify,
                    number=number,
                    repeat=current_repeat,
                    warmup_runs=warmup_runs
                )
                
                # Stop bandwidth profiling and get statistics
                if bandwidth_connected and bandwidth_client:
                    bandwidth_client.stop()
                    print("✓ Stopped bandwidth profiling")
                    
                    # Get all data and calculate statistics
                    try:
                        data = bandwidth_client.get_data()
                        if data and len(data) > 0:
                            # Get statistics for the entire period
                            start_ns = min(d.timestamp_ns for d in data)
                            end_ns = max(d.timestamp_ns for d in data)
                            bandwidth_stats = bandwidth_client.get_statistics(start_ns, end_ns)
                            
                            # Check if bandwidth values are zero or invalid
                            has_valid_bandwidth = (
                                bandwidth_stats.read_avg_mbps > 0 or
                                bandwidth_stats.write_avg_mbps > 0 or
                                bandwidth_stats.total_avg_mbps > 0
                            )
                            
                            if not has_valid_bandwidth and retry_attempt < max_bandwidth_retries:
                                print(f"⚠ Bandwidth values are zero, increasing repeat from {current_repeat} to {current_repeat * 2} and retrying...")
                                current_repeat = current_repeat * 2
                                continue  # Retry with increased repeat
                            
                            # Add bandwidth statistics to result
                            result["bandwidth_read_avg_mbps"] = bandwidth_stats.read_avg_mbps
                            result["bandwidth_read_max_mbps"] = bandwidth_stats.read_max_mbps
                            result["bandwidth_write_avg_mbps"] = bandwidth_stats.write_avg_mbps
                            result["bandwidth_write_max_mbps"] = bandwidth_stats.write_max_mbps
                            result["bandwidth_total_avg_mbps"] = bandwidth_stats.total_avg_mbps
                            result["bandwidth_total_max_mbps"] = bandwidth_stats.total_max_mbps
                            result["bandwidth_sample_count"] = bandwidth_stats.sample_count
                            result["bandwidth_duration_sec"] = bandwidth_stats.duration_sec
                            
                            if has_valid_bandwidth:
                                print(f"✓ Bandwidth stats: Read {bandwidth_stats.read_avg_mbps:.1f} MB/s (max {bandwidth_stats.read_max_mbps:.1f}), "
                                      f"Write {bandwidth_stats.write_avg_mbps:.1f} MB/s (max {bandwidth_stats.write_max_mbps:.1f}), "
                                      f"Total {bandwidth_stats.total_avg_mbps:.1f} MB/s (max {bandwidth_stats.total_max_mbps:.1f})")
                            else:
                                print("⚠ Bandwidth values are zero after retries")
                        else:
                            if retry_attempt < max_bandwidth_retries:
                                print(f"⚠ No bandwidth data collected, increasing repeat from {current_repeat} to {current_repeat * 2} and retrying...")
                                current_repeat = current_repeat * 2
                                continue  # Retry with increased repeat
                            
                            print("⚠ No bandwidth data collected")
                            # Set empty values so CSV columns are populated
                            result["bandwidth_read_avg_mbps"] = None
                            result["bandwidth_read_max_mbps"] = None
                            result["bandwidth_write_avg_mbps"] = None
                            result["bandwidth_write_max_mbps"] = None
                            result["bandwidth_total_avg_mbps"] = None
                            result["bandwidth_total_max_mbps"] = None
                            result["bandwidth_sample_count"] = None
                            result["bandwidth_duration_sec"] = None
                    except Exception as e:
                        if retry_attempt < max_bandwidth_retries:
                            print(f"⚠ Failed to get bandwidth statistics: {e}, increasing repeat and retrying...")
                            current_repeat = current_repeat * 2
                            continue
                        
                        print(f"⚠ Failed to get bandwidth statistics: {e}")
                        # Set empty values on error
                        result["bandwidth_read_avg_mbps"] = None
                        result["bandwidth_read_max_mbps"] = None
                        result["bandwidth_write_avg_mbps"] = None
                        result["bandwidth_write_max_mbps"] = None
                        result["bandwidth_total_avg_mbps"] = None
                        result["bandwidth_total_max_mbps"] = None
                        result["bandwidth_sample_count"] = None
                        result["bandwidth_duration_sec"] = None
                else:
                    # No bandwidth profiling - set empty values
                    result["bandwidth_read_avg_mbps"] = None
                    result["bandwidth_read_max_mbps"] = None
                    result["bandwidth_write_avg_mbps"] = None
                    result["bandwidth_write_max_mbps"] = None
                    result["bandwidth_total_avg_mbps"] = None
                    result["bandwidth_total_max_mbps"] = None
                    result["bandwidth_sample_count"] = None
                    result["bandwidth_duration_sec"] = None
                
                # If we got here, we either have valid bandwidth or no bandwidth profiling
                break
                
            except Exception as e:
                # If run_so fails, don't retry
                if result is None:
                    result = {"success": False, "error": str(e)}
                break
    finally:
        # Cleanup bandwidth client
        if bandwidth_client:
            try:
                bandwidth_client.disconnect()
            except:
                pass
    
    # Check if timeout occurred during run_so
    if timeout_occurred.is_set() and result.get("success"):
        return {
            "success": False,
            "error": "Operation timed out"
        }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Load and run kernel artifacts from tuning logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options
    parser.add_argument("--artifact", type=str, help="Path to a specific artifact (.tar or trace_*.py)")
    parser.add_argument("--dir", type=str, help="Directory containing artifacts")
    
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
                        choices=["opencl", "vulkan", "cpu", "llvm"],
                        help="Device backend. Auto-detected from filename if omitted. 'llvm' is alias for 'cpu'.")
    
    # Benchmark options
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip correctness verification")
    parser.add_argument("--number", type=int, default=10,
                        help="Number of times to run for each measurement (default: 10)")
    parser.add_argument("--repeat", type=int, default=20,
                        help="Number of measurements to take (default: 20)")
    parser.add_argument("--warmup-runs", type=int, default=5,
                        help="Number of warm-up runs (default: 5)")
    
    # Timeout options
    parser.add_argument("--timeout", type=int, default=600,
                        help="Overall timeout in seconds per artifact (default: 600)")
    
    # Bandwidth profiler options
    parser.add_argument("--bandwidth-host", type=str, default=None,
                        help="GPU bandwidth profiler host (enables bandwidth measurement for GPU backends)")
    parser.add_argument("--bandwidth-port", type=int, default=8888,
                        help="GPU bandwidth profiler port (default: 8888)")
    
    # Output options
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV output file path (auto-generated if not provided)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of artifacts to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.artifact and not args.dir:
        parser.error("Must provide either --artifact or --dir")
    
    # Collect artifacts
    artifacts = []
    if args.artifact:
        if not os.path.exists(args.artifact):
            print(f"Error: Artifact not found: {args.artifact}")
            return
        artifacts = [args.artifact]
    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"Error: Directory not found: {args.dir}")
            return
        
        # Find all artifacts
        tar_files = glob.glob(os.path.join(args.dir, "*.tar"))
        so_files = glob.glob(os.path.join(args.dir, "*.so"))
        trace_files = glob.glob(os.path.join(args.dir, "trace_*.py"))
        
        artifacts = tar_files + so_files + trace_files
        artifacts.sort()
        
        total_artifacts = len(artifacts)
        if args.limit:
            artifacts = artifacts[:args.limit]
            print(f"Found {total_artifacts} artifacts, limiting to {len(artifacts)} for testing")
        else:
            print(f"Found {len(artifacts)} artifacts")
    
    # Determine shape if provided
    shape = None
    if args.m and args.k and args.n:
        shape = (args.m, args.k, args.n)
    
    # Normalize backend: "llvm" -> "cpu"
    backend = args.backend
    if backend == "llvm":
        backend = "cpu"
    
    # Prepare CSV output - always include bandwidth columns for GPU backends
    # (bandwidth profiling is auto-enabled for opencl/vulkan backends)
    csv_header = ["filename", "m", "k", "n", "backend",
                  "mean_ms", "std_ms", "min_ms", "max_ms", "success", "error"]
    # Always include bandwidth columns (will be empty if not available)
    csv_header.extend([
        "bandwidth_read_avg_mbps", "bandwidth_read_max_mbps",
        "bandwidth_write_avg_mbps", "bandwidth_write_max_mbps",
        "bandwidth_total_avg_mbps", "bandwidth_total_max_mbps",
        "bandwidth_sample_count", "bandwidth_duration_sec"
    ])
    
    # Determine CSV output path
    csv_path = args.csv
    if not csv_path and args.dir:
        # Auto-generate CSV filename based on directory and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(args.dir, f"bench_{timestamp}.csv")
    elif not csv_path and args.artifact:
        # Auto-generate CSV filename based on artifact directory
        artifact_dir = os.path.dirname(args.artifact)
        if artifact_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(artifact_dir, f"bench_{timestamp}.csv")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"bench_{timestamp}.csv"
    
    # Initialize CSV file with header if it doesn't exist
    csv_file_exists = False
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        csv_file_exists = os.path.exists(csv_path)
        if not csv_file_exists:
            # Write header for new file
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)
            print(f"✓ Created CSV file: {csv_path}")
        else:
            print(f"✓ Appending to existing CSV file: {csv_path}")
    
    # Format values for CSV (defined once outside loop)
    def format_float(value, decimals=3):
        if value is None:
            return ""
        try:
            return round(float(value), decimals)
        except (ValueError, TypeError):
            return ""
    
    def format_int(value):
        if value is None:
            return ""
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return ""
    
    # Run each artifact
    results = []
    skipped_count = 0
    for idx, artifact_path in enumerate(artifacts, 1):
        print(f"\n[{idx}/{len(artifacts)}] Processing: {os.path.basename(artifact_path)}")
        
        # Determine backend for this artifact
        artifact_backend = parse_backend_from_filename(artifact_path) or backend or "opencl"
        
        result = run_artifact(
            artifact_path,
            shape=shape,
            tracker_host=args.tracker_host,
            tracker_port=args.tracker_port,
            tracker_key=args.tracker_key,
            backend=backend,
            verify=not args.no_verify,
            number=args.number,
            repeat=args.repeat,
            warmup_runs=args.warmup_runs,
            timeout_sec=args.timeout,
            bandwidth_host=args.bandwidth_host,
            bandwidth_port=args.bandwidth_port
        )
        
        # Skip failed artifacts for OpenCL backend (invalid kernels)
        if not result.get("success") and artifact_backend == "opencl":
            print(f"⚠ Skipping failed OpenCL artifact (invalid kernel)")
            skipped_count += 1
            continue  # Skip this artifact, don't add to CSV
        
        results.append((artifact_path, result))
        
        # Parse shape and backend for CSV
        artifact_shape = parse_shape_from_filename(artifact_path) or shape
        artifact_backend = parse_backend_from_filename(artifact_path) or backend or "unknown"
        artifact_m = artifact_shape[0] if artifact_shape else ""
        artifact_k = artifact_shape[1] if artifact_shape else ""
        artifact_n = artifact_shape[2] if artifact_shape else ""
        
        # Build CSV row
        csv_row = [
            os.path.basename(artifact_path),
            artifact_m,
            artifact_k,
            artifact_n,
            artifact_backend,
            format_float(result.get("mean_ms")),
            format_float(result.get("std_ms")),
            format_float(result.get("min_ms")),
            format_float(result.get("max_ms")),
            result.get("success"),
            result.get("error", ""),
        ]
        # Always include bandwidth columns (will be empty if not available)
        csv_row.extend([
            format_float(result.get("bandwidth_read_avg_mbps"), decimals=1),
            format_float(result.get("bandwidth_read_max_mbps"), decimals=1),
            format_float(result.get("bandwidth_write_avg_mbps"), decimals=1),
            format_float(result.get("bandwidth_write_max_mbps"), decimals=1),
            format_float(result.get("bandwidth_total_avg_mbps"), decimals=1),
            format_float(result.get("bandwidth_total_max_mbps"), decimals=1),
            format_int(result.get("bandwidth_sample_count")),
            format_float(result.get("bandwidth_duration_sec"), decimals=3)
        ])
        
        # Append row to CSV file immediately after each artifact
        if csv_path:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_row)
            print(f"✓ Results appended to: {csv_path}")
    
    # Print summary
    successful = sum(1 for _, r in results if r.get("success"))
    failed = len(results) - successful
    print(f"\n{'='*60}")
    print(f"Summary: {successful} successful, {failed} failed, {skipped_count} skipped (invalid kernels) out of {len(artifacts)} artifacts")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

