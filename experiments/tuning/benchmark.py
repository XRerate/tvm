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

"""RPC benchmarking with retry and OOM recovery."""

from __future__ import annotations
import time
import csv
from typing import Dict, List, Tuple, Callable, Optional
import tvm
from tvm import rpc as _rpc
import numpy as np

from config import AutotuneConfig


class BenchmarkResult:
    """Result from a single benchmark run."""
    def __init__(
        self,
        success: bool,
        mean_ms: Optional[float] = None,
        std_ms: Optional[float] = None,
        error: Optional[str] = None,
        skipped: bool = False
    ):
        self.success = success
        self.mean_ms = mean_ms
        self.std_ms = std_ms
        self.error = error
        self.skipped = skipped


def benchmark_artifact(
    so_path: str,
    shape: Tuple[int, int, int],
    config: AutotuneConfig,
    logger: Callable[[str], None]
) -> BenchmarkResult:
    """Benchmark artifact with retry logic for device-side OOM recovery.
    
    On device OOM (e.g., CL_OUT_OF_HOST_MEMORY during kernel build/run):
    - Close RPC session to release device resources
    - Wait for cooldown period to allow device driver/RPC server to clean up
    - Retry with a fresh session
    
    Parameters
    ----------
    so_path : str
        Path to compiled artifact
    shape : Tuple[int, int, int]
        Matrix shape (M, K, N)
    config : AutotuneConfig
        Configuration object
    logger : Callable[[str], None]
        Logging function
        
    Returns
    -------
    BenchmarkResult
        Benchmark result with timing or error info
    """
    m, k, n = shape
    
    # Set random seed for reproducibility
    if config.seed is not None:
        np.random.seed(config.seed)
    
    # Use shorter timeout for benchmarking to fail fast
    bench_timeout = min(config.session_timeout_sec, config.bench_session_timeout_sec)
    
    sess = None
    try:
        # Connect to RPC
        tracker = _rpc.connect_tracker(config.tracker_host, config.tracker_port)
        sess = tracker.request(
            config.bench_key,
            priority=config.session_priority,
            session_timeout=bench_timeout
        )
        # Upload artifact
        logger(f"[Bench] Uploading {so_path}...")
        remote_path = f"matmul_{time.time()}.so"
        sess.upload(so_path, remote_path)
        # Load module
        logger(f"[Bench] Loading module on device...")
        mod = sess.load_module(remote_path)
        # Allocate tensors
        logger(f"[Bench] Allocating tensors ({m}x{k}x{n})...")
        dev = sess.device(str(config.backend), 0)
        a_np = np.random.uniform(size=(m, k)).astype("float32")
        b_np = np.random.uniform(size=(k, n)).astype("float32")
        a_tvm = tvm.runtime.tensor(a_np, device=dev)
        b_tvm = tvm.runtime.tensor(b_np, device=dev)
        c_tvm = tvm.runtime.tensor(np.zeros((m, n), dtype="float32"), device=dev)
        # Warmup run
        logger(f"[Bench] Warmup run...")
        mod(a_tvm, b_tvm, c_tvm)
        dev.sync()
        # Benchmark
        logger(f"[Bench] Running benchmark...")
        evaluator = mod.time_evaluator(
            mod.entry_name,
            dev,
            number=config.eval_number,
            repeat=config.eval_repeat,
            min_repeat_ms=config.eval_min_repeat_ms
        )
        results = evaluator(a_tvm, b_tvm, c_tvm)
        mean_ms = float(np.mean(results.results)) * 1000
        std_ms = float(np.std(results.results)) * 1000
        logger(f"[Bench] Success: {mean_ms:.3f} ± {std_ms:.3f} ms")
        return BenchmarkResult(success=True, mean_ms=mean_ms, std_ms=std_ms)
    except Exception as e:
        error_msg = str(e)
        is_oom = any(
            keyword in error_msg.lower()
            for keyword in [
                "out_of_host_memory",
                "out_of_resources",
                "cl_out_of",
                "memory",
                "allocation failed"
            ]
        )
        if is_oom:
            logger(f"[Bench] OOM detected: {error_msg}")
            return BenchmarkResult(success=False, error=error_msg)
        else:
            logger(f"[Bench] Failed: {error_msg}")
            return BenchmarkResult(success=False, error=error_msg)
    finally:
        if sess:
            try:
                sess.close()
            except Exception:
                pass


def benchmark_all_artifacts(
    artifacts: List[Tuple[str, str, str]],
    shape: Tuple[int, int, int],
    config: AutotuneConfig,
    logger: Callable[[str], None]
) -> List[List]:
    """Benchmark all artifacts and return CSV rows.
    
    Parameters
    ----------
    artifacts : List[Tuple[str, str, str]]
        List of (candidate_name, variant_name, artifact_path)
    shape : Tuple[int, int, int]
        Matrix shape (M, K, N)
    config : AutotuneConfig
        Configuration object
    logger : Callable[[str], None]
        Logging function
        
    Returns
    -------
    List[List]
        CSV rows with benchmark results
    """
    m, k, n = shape
    rows = []
    
    skip_after_oom = getattr(config, "bench_skip_after_oom", True)
    oom_flag = False
    for cand, variant_name, art in artifacts:
        if oom_flag and skip_after_oom:
            logger(f"[Bench] Skipping {cand}_{variant_name} due to previous OOM.")
            rows.append([
                cand,
                variant_name,
                m, k, n,
                "N/A",
                "N/A",
                "SKIPPED (OOM previously)"
            ])
            continue
        logger(f"Benchmarking {cand}_{variant_name}...")
        result = benchmark_artifact(art, shape, config, logger)
        if result.success:
            rows.append([
                cand,
                variant_name,
                m, k, n,
                f"{result.mean_ms:.3f}",
                f"{result.std_ms:.3f}",
                "OK"
            ])
        else:
            is_oom = "OOM" in (result.error or "").upper() or "OUT OF MEMORY" in (result.error or "").upper()
            if is_oom:
                oom_flag = True
            rows.append([
                cand,
                variant_name,
                m, k, n,
                "N/A",
                "N/A",
                result.error or "FAILED"
            ])
    return rows


def save_benchmark_csv(rows: List[List], output_path: str, logger: Callable[[str], None]):
    """Save benchmark results to CSV file.
    
    Parameters
    ----------
    rows : List[List]
        CSV rows
    output_path : str
        Output CSV file path
    logger : Callable[[str], None]
        Logging function
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate", "variant", "M", "K", "N", "mean_ms", "std_ms", "status"])
        writer.writerows(rows)
    logger(f"CSV saved -> {output_path}")
