"""
Reproduce and benchmark schedules from saved trace files.

Usage:
    # Default: GPU OpenCL with RPC benchmarking
    conda activate tvm-build-venv
    python3 repro.py
    
    # GPU Vulkan
    TVM_GPU_BACKEND=vulkan python3 repro.py
    
    # CPU
    DEVICE_TYPE=cpu python3 repro.py
    
    # Build only (no benchmark)
    SKIP_BENCHMARK=1 python3 repro.py

Requirements:
    - Update trace_file path in main() function
    - Ensure RPC tracker and device are running (unless SKIP_BENCHMARK=1)
    - For GPU: Use separate RPC key 'android64_bench' for benchmarking
    - For CPU: Use RPC key 'android64'
"""

import tvm
from tvm import tir, te, rpc
from tvm.contrib import ndk
import os
import numpy as np
import tempfile

def get_matmul_workload(M, N, K, dtype="float32"):
    """Return the TE placeholders describing the matmul workload.

    The order (A[M,K], B[K,N]) with reduction axis k is canonical for GEMM.
    We do NOT apply manual scheduling here; MetaSchedule will explore transformations.
    Returning raw tensors keeps the search space flexible (tiling, fusion, vectorization...).
    """
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    return [A, B, C]


def load_schedule_from_trace_file(trace_file, workload_mod, func_name="main"):
    """Load a schedule from a trace file saved by str(rec.trace).
    
    Parameters
    ----------
    trace_file : str
        Path to the trace file (Python code format).
    workload_mod : tvm.IRModule
        The original workload module (before scheduling).
    func_name : str
        The function name in the module (default: "main").
        Must match the func_name in the trace file.
    
    Returns
    -------
    tir.Schedule
        The schedule object with the trace applied.
    """
    # Create schedule from the module
    sch = tir.Schedule(workload_mod, debug_mask="all")
    
    # Load and apply the trace from file
    # The file contains Python code that defines apply_trace function
    with open(trace_file, 'r') as f:
        trace_code = f.read()
    
    # Execute the trace code to define apply_trace function
    exec(trace_code, globals())
    
    # Apply the trace to the schedule
    apply_trace(sch)
    
    return sch


def build_and_benchmark_schedule(sch, m, n, k, backend="opencl", use_rpc=True, device_type="gpu"):
    """Build the scheduled module and benchmark it.
    
    Parameters
    ----------
    sch : tir.Schedule
        The schedule to build and benchmark.
    m, n, k : int
        Matrix dimensions.
    backend : str
        Backend to use: "opencl", "vulkan" (for GPU), or "llvm" (for CPU).
    use_rpc : bool
        If True, benchmark on remote device via RPC.
        If False, just build locally (for testing).
    device_type : str
        "gpu" or "cpu" - determines target and benchmarking configuration.
    
    Returns
    -------
    dict
        Benchmark results with mean_ms, std_ms, min_ms, max_ms.
    """
    # Build the module
    print("\n" + "=" * 80)
    print("Building scheduled module...")
    print("=" * 80)
    
    host_triple = "llvm -mtriple=aarch64-linux-android"
    
    if device_type == "cpu":
        # CPU target
        target = tvm.target.Target(f"{host_triple}")
    else:
        # GPU target
        if backend == "vulkan":
            target = tvm.target.Target("vulkan", host=host_triple)
        else:
            target = tvm.target.Target("opencl -device=adreno", host=host_triple)
    
    try:
        rt_mod = tvm.build(sch.mod, target=target)
        print(f"✓ Build successful (target: {target})")
    except Exception as e:
        print(f"✗ Build failed: {e}")
        return None
    
    if not use_rpc:
        print("Skipping benchmark (use_rpc=False)")
        return None
    
    # Export as .so
    lib_path = os.path.join(tempfile.mkdtemp(), "repro_matmul.so")
    try:
        rt_mod.export_library(lib_path, fcompile=ndk.create_shared)
        print(f"✓ Exported to: {lib_path}")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return None
    
    # Connect to RPC
    print("\n" + "=" * 80)
    print("Connecting to RPC for benchmarking...")
    print("=" * 80)
    
    tracker_host = "127.0.0.1"
    tracker_port = 9190
    
    # Use appropriate RPC key based on device type
    if device_type == "cpu":
        tracker_key = "android64"  # CPU can use single key
    else:
        tracker_key = "android64_bench"  # GPU uses separate benchmark key
    
    try:
        tracker = rpc.connect_tracker(tracker_host, tracker_port)
        remote = tracker.request(tracker_key, session_timeout=300, priority=1)
        print(f"✓ Connected to remote device")
    except Exception as e:
        print(f"✗ RPC connection failed: {e}")
        print(f"  Make sure RPC tracker and device are running with key '{tracker_key}'")
        return None
    
    # Upload and load module
    try:
        remote.upload(lib_path)
        remote_filename = os.path.basename(lib_path)
        remote_mod = remote.load_module(remote_filename)
        print(f"✓ Module loaded on device")
    except Exception as e:
        print(f"✗ Module loading failed: {e}")
        del remote
        return None
    
    # Allocate device tensors
    try:
        if device_type == "cpu":
            rdev = remote.cpu()
        elif backend == "vulkan":
            try:
                rdev = remote.vulkan(0)
            except:
                print("[WARN] Vulkan failed, falling back to OpenCL")
                rdev = remote.cl(0)
        else:
            rdev = remote.cl(0)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        a_np = np.random.uniform(size=(m, k)).astype(np.float32)
        b_np = np.random.uniform(size=(k, n)).astype(np.float32)
        
        ra = tvm.runtime.tensor(a_np, rdev)
        rb = tvm.runtime.tensor(b_np, rdev)
        rc = tvm.runtime.tensor(np.zeros((m, n), dtype=np.float32), rdev)
        print(f"✓ Device tensors allocated ({device_type})")
    except Exception as e:
        print(f"✗ Device allocation failed: {e}")
        del remote
        return None
    
    # Get entry function
    try:
        r_entry = "main"
        r_f = remote_mod[r_entry]
    except Exception as e:
        print(f"✗ Entry function lookup failed: {e}")
        del remote
        return None
    
    # Warmup and verify correctness
    print("\n" + "=" * 80)
    print("Running warmup and correctness check...")
    print("=" * 80)
    
    try:
        r_f(ra, rb, rc)
        result = rc.numpy()
        expected = np.dot(a_np, b_np)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
        print("✓ Correctness check passed")
    except Exception as e:
        print(f"✗ Correctness check failed: {e}")
        del remote
        return None
    
    # Benchmark
    print("\n" + "=" * 80)
    print("Benchmarking...")
    print("=" * 80)
    
    try:
        time_f = remote_mod.time_evaluator(r_entry, rdev, number=10, repeat=20)
        timing_result = time_f(ra, rb, rc)
        
        times_s = timing_result.results
        mean_ms = float(np.mean(times_s)) * 1000.0
        std_ms = float(np.std(times_s)) * 1000.0
        min_ms = float(np.min(times_s)) * 1000.0
        max_ms = float(np.max(times_s)) * 1000.0
        
        print(f"✓ Benchmark completed:")
        print(f"  Mean: {mean_ms:.3f} ms")
        print(f"  Std:  {std_ms:.3f} ms")
        print(f"  Min:  {min_ms:.3f} ms")
        print(f"  Max:  {max_ms:.3f} ms")
        
        del remote
        
        return {
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
        }
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        del remote
        return None


def main():
    """Example usage: load a schedule from a trace file and benchmark it."""
    
    # Configuration
    trace_file = "/home/seonjunkim/projects/tvm/tuning_logs/20251111_131247/trace_256x256x256_cand001.py"
    
    # Device type: "gpu" (default) or "cpu"
    device_type = os.environ.get("DEVICE_TYPE", "gpu").lower()
    
    # Backend: for GPU = "opencl" (default) or "vulkan", for CPU = "llvm"
    if device_type == "cpu":
        backend = "llvm"
    else:
        backend = os.environ.get("TVM_GPU_BACKEND", "opencl").lower()
    
    # Skip RPC benchmark (just build and test locally)
    use_rpc = os.environ.get("SKIP_BENCHMARK", "0") != "1"
    
    if not os.path.exists(trace_file):
        print(f"Error: Trace file not found: {trace_file}")
        print(f"Please update the trace_file path in repro.py")
        return
    
    # Create the original workload
    m, n, k = 256, 256, 256
    A, B, C = get_matmul_workload(m, n, k)
    
    # IMPORTANT: Use "main" as global_symbol to match the trace file
    # The trace file has func_name="main", so the module must have the same name
    func = te.create_prim_func([A, B, C]).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    
    # Load the schedule from trace file
    print("=" * 80)
    print("Loading schedule from trace file...")
    print("=" * 80)
    print(f"Trace file: {trace_file}")
    print(f"Device type: {device_type}")
    print(f"Backend: {backend}")
    
    sch = load_schedule_from_trace_file(trace_file, mod, func_name="main")
    print("✓ Schedule loaded successfully")
    
    # Print the resulting scheduled module (optional, can be commented out)
    print("\n" + "=" * 80)
    print("Scheduled TIR (first 50 lines):")
    print("=" * 80)
    script_lines = sch.mod["main"].script().split('\n')
    print('\n'.join(script_lines[:50]))
    if len(script_lines) > 50:
        print(f"... ({len(script_lines) - 50} more lines)")
    
    # Build and benchmark
    if use_rpc:
        print(f"\nNote: To skip benchmarking, set SKIP_BENCHMARK=1")
        print(f"Note: To use CPU instead of GPU, set DEVICE_TYPE=cpu")
    
    results = build_and_benchmark_schedule(
        sch, m, n, k, 
        backend=backend, 
        use_rpc=use_rpc, 
        device_type=device_type
    )
    
    if results:
        print("\n" + "=" * 80)
        print("SUCCESS: Schedule reproduced and benchmarked")
        print("=" * 80)
        print(f"Performance: {results['mean_ms']:.3f} ± {results['std_ms']:.3f} ms")
        print(f"Min: {results['min_ms']:.3f} ms, Max: {results['max_ms']:.3f} ms")
    elif use_rpc:
        print("\n" + "=" * 80)
        print("FAILED: Could not benchmark schedule")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Build completed (benchmarking skipped)")
        print("=" * 80)


if __name__ == "__main__":
    main()

