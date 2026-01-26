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

"""Tuning orchestrator that coordinates all components."""

from __future__ import annotations
import time
import logging
from typing import Tuple, List, Callable
import tvm
import tvm.ir
from tvm import te, meta_schedule as ms
from tvm.meta_schedule.runner import RPCRunner
from tvm.meta_schedule.runner.utils import run_evaluator_bw
from tvm.meta_schedule.runner.config import RPCConfig, EvaluatorConfig
from tvm.meta_schedule.builder import LocalBuilder
from tvm.contrib.hexagon.meta_schedule import get_hexagon_rpc_runner
from tvm.contrib.hexagon.build import HexagonLauncherAndroid

from config import AutotuneConfig
from targets import verify_codegen_available
from artifacts import build_all_variants, create_android_export_function, create_hexagon_export_function
from space import create_hexagon_space, create_schedule_space, get_schedule_rules_info


def get_matmul_workload(m: int, n: int, k: int, dtype: str = "float32"):
    """Create TE matmul workload.
    
    Parameters
    ----------
    m, n, k : int
        Matrix dimensions
    dtype : str
        Data type
        
    Returns
    -------
    List
        [A, B, C] tensor placeholders/compute
    """
    A = te.placeholder((m, k), name="A", dtype=dtype)
    B = te.placeholder((k, n), name="B", dtype=dtype)
    k_axis = te.reduce_axis((0, k), name="k")
    C = te.compute((m, n), lambda i, j: te.sum(A[i, k_axis] * B[k_axis, j], axis=k_axis), name="C")
    return [A, B, C]


def create_runner(config: AutotuneConfig) -> RPCRunner:
    """Create RPCRunner from config.
    
    Parameters
    ----------
    config : AutotuneConfig
        Configuration object
        
    Returns
    -------
    RPCRunner
        Configured RPC runner
    """
    if config.backend == "hexagon":
        hexagon_launcher = HexagonLauncherAndroid(
                serial_number="R3CX80PSH7N",
                rpc_info={
                    "tracker_host": config.tracker_host,
                    "tracker_port": config.tracker_port,
                    "device_key": config.tracker_key,
                    "session_timeout_sec": config.session_timeout_sec,
                    "session_priority": config.session_priority,
                    "workspace_base": "/data/local/tmp/hexagon_workspace",
                    "adb_server_socket": "tcp:5037",
                },
            )
        return get_hexagon_rpc_runner(
            hexagon_launcher=hexagon_launcher,
            number=config.eval_number,
            repeat=config.eval_repeat,
            min_repeat_ms=config.eval_min_repeat_ms,
            max_workers=1,
        )
    else:
        def get_run_evaluator_bw(session, rt_mod, device, evaluator_config, repeated_args):
            return run_evaluator_bw(rt_mod, device, evaluator_config, repeated_args)
        
        return RPCRunner(
            rpc_config=RPCConfig(
                tracker_host=config.tracker_host,
                tracker_port=config.tracker_port,
                tracker_key=config.tracker_key,
                session_timeout_sec=config.session_timeout_sec,
                session_priority=config.session_priority,
            ),
            evaluator_config=EvaluatorConfig(
                number=config.eval_number,
                repeat=config.eval_repeat,
                min_repeat_ms=config.eval_min_repeat_ms,
                enable_cpu_cache_flush=config.eval_enable_cpu_cache_flush
            ),
            max_workers=1,
            f_run_evaluator=get_run_evaluator_bw,
        )


def create_builder(config: AutotuneConfig, logger: Callable[[str], None]) -> LocalBuilder:
    """Create LocalBuilder from config.
    
    Parameters
    ----------
    config : AutotuneConfig
        Configuration object
    logger : Callable[[str], None]
        Logging function
        
    Returns
    -------
    LocalBuilder
        Configured local builder
    """
    if config.backend == "hexagon":
        f_export = create_hexagon_export_function(logger)
    else:
        f_export = create_android_export_function(logger)
    return LocalBuilder(
        max_workers=1,
        timeout_sec=config.builder_timeout_sec,
        f_export=f_export
    )


def tune_shape(
    shape: Tuple[int, int, int],
    config: AutotuneConfig,
    remote_target: tvm.target.Target,
    work_dir: str,
    builder: LocalBuilder,
    runner: RPCRunner,
    logger: Callable[[str], None],
    strategy: str,
    database: str
) -> List:
    """Tune a single matrix shape.
    
    Parameters
    ----------
    shape : Tuple[int, int, int]
        Matrix shape (M, K, N)
    config : AutotuneConfig
        Configuration object
    remote_target : tvm.target.Target
        Remote target
    work_dir : str
        Working directory
    builder : LocalBuilder
        Builder instance
    runner : RPCRunner
        Runner instance
    logger : Callable[[str], None]
        Logging function
        
    Returns
    -------
    List
        Tuning records for this shape
    """
    m, k, n = shape
    
    # Build TE IRModule for this shape
    A, B, C = get_matmul_workload(m, n, k)
    func = te.create_prim_func([A, B, C]).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    
    logger(f"Tuning start: shape={m}x{k}x{n}, target={remote_target}, trials={config.trials}")
    logger(f"Timeouts: build={config.builder_timeout_sec}s, session={config.session_timeout_sec}s")
    logger(f"Note: Large matmuls may take several minutes per trial.")
    logger(f"Each trial: compile (~30s-2min) + RPC upload + device execution (~10s-1min)")
    
    # Create custom schedule space if configured
    space = create_schedule_space(config)
    if space is not None:
        space_info = get_schedule_rules_info(config)
        logger(f"Using custom schedule space:\n{space_info}")
    else:
        logger(f"Using default schedule rules for {config.backend}")
    
    # Enable verbose TVM logging
    tvm_logger = logging.getLogger("tvm")
    original_level = tvm_logger.level
    tvm_logger.setLevel(logging.DEBUG)

    latency_cost_model = "xgb"
    bandwidth_cost_model = None

    if strategy == "nsgaii":
        if database != "json_pareto":
            logger(f"Warning: nsgaii strategy requires json_pareto database")
            database = "json_pareto"
        bandwidth_cost_model = "xgb_bw"
    
    # Set random seed for reproducibility
    if config.seed is not None:
        import random
        import numpy as np
        random.seed(config.seed)
        np.random.seed(config.seed)
        logger(f"Random seed set to {config.seed} for reproducibility")
    
    t0 = time.monotonic()
    try:
        if space != None:
            db = ms.tune_tir(
                mod=mod,
                target=remote_target,
                work_dir=work_dir,
                max_trials_global=config.trials,
                latency_cost_model=latency_cost_model,
                bandwidth_cost_model=bandwidth_cost_model,
                builder=builder,
                runner=runner,
                num_trials_per_iter=config.num_trials_per_iter,
                space=space,  # Use custom space if available
                strategy=strategy,
                database=database,
                seed=config.seed,
            )
        else:
            db = ms.tune_tir(
                mod=mod,
                target=remote_target,
                work_dir=work_dir,
                max_trials_global=config.trials,
                latency_cost_model=latency_cost_model,
                bandwidth_cost_model=bandwidth_cost_model,
                builder=builder,
                runner=runner,
                num_trials_per_iter=config.num_trials_per_iter,
                strategy=strategy,
                database=database,
                seed=config.seed,
            )
    except Exception as e:
        logger(f"Tuning failed: {e}")
        raise
    finally:
        tvm_logger.setLevel(original_level)
    
    # Filter records to only include those for the current workload (shape)
    all_records = list(db.get_all_tuning_records())
    mod_hash = tvm.ir.structural_hash(mod)
    records = [rec for rec in all_records if tvm.ir.structural_hash(rec.workload.mod) == mod_hash]
    logger(f"Tuning done in {time.monotonic()-t0:.1f}s. Records: {len(records)} (filtered from {len(all_records)} total)")
    
    return records


def process_shape(
    shape: Tuple[int, int, int],
    config: AutotuneConfig,
    remote_target: tvm.target.Target,
    work_dir: str,
    builder: LocalBuilder,
    runner: RPCRunner,
    logger: Callable[[str], None]
) -> List[str]:
    """Process a single shape: tune, build, and benchmark.
    
    Parameters
    ----------
    shape : Tuple[int, int, int]
        Matrix shape (M, K, N)
    config : AutotuneConfig
        Configuration object
    remote_target : tvm.target.Target
        Remote target
    work_dir : str
        Working directory
    builder : LocalBuilder
        Builder instance
    runner : RPCRunner
        Runner instance
    logger : Callable[[str], None]
        Logging function
        
    Returns
    -------
    List[str]
        List of artifact paths
    """
    m, k, n = shape
    shape_name = f"{m}x{k}x{n}"
    
    # Tune
    records = tune_shape(shape, config, remote_target, work_dir, builder, runner, logger, config.strategy, config.database)
    
    # Verify codegen availability
    ok, msg = verify_codegen_available(config.backend)
    if not ok:
        logger(f"[Warning] {msg}")
        logger(f"[Warning] Skipping artifact builds for {shape_name}")
        return []
    
    # Build all candidates with all variants
    all_artifacts: List[Tuple[str, str, str]] = []
    for idx, rec in enumerate(records, start=1):
        artifacts = build_all_variants(
            record=rec,
            idx=idx,
            shape_name=shape_name,
            config=config,
            base_target=remote_target,
            work_dir=work_dir,
            logger=logger
        )
        all_artifacts.extend(artifacts)
    
    # Benchmark all artifacts
    '''
    logger(f"Benchmarking exported artifacts via RPC (key={config.bench_key})...")
    rows = benchmark_all_artifacts(all_artifacts, shape, config, logger)
    
    # Save CSV
    csv_path = os.path.join(work_dir, f"bench_{shape_name}_{config.backend}.csv")
    save_benchmark_csv(rows, csv_path, logger)
    '''
    
    return [art for _, _, art in all_artifacts]
