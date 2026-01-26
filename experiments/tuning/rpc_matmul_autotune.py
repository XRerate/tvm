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
Minimal RPC autotune for TE matmul
=================================================================

Goals (trimmed to essentials):
  - Define a TE matmul workload (no manual schedule).
  - Run MetaSchedule tune_tir on a remote target via RPC.
  - Reconstruct the best schedule, build, and export a .so (or .tar fallback).
  - Save artifacts under ./tuning_logs/<timestamp>/.

Programmatic API:
    run_autotune(shape=(64,64,64), trials=3, work_dir=None, backend=None, export="so", config_path=None)
    -> returns {"work_dir": str, "artifacts": [str], "log": str}

CLI:
  python rpc_matmul_autotune.py --config opencl.toml [--backend opencl|vulkan|hexagon] [--trials 3]
"""

from __future__ import annotations
import argparse
import datetime
import io
import os
import sys
from typing import Dict, List, Optional

from config import load_config, save_effective_config
from targets import get_targets
from tuner import create_runner, create_builder, process_shape


def _ts() -> str:
    """Get current timestamp string."""
    return datetime.datetime.now().strftime("%H:%M:%S")


def run_autotune(
    shape: tuple[int, int, int] | None = None,
    trials: int | None = None,
    work_dir: str | None = None,
    backend: str | None = None,
    export: str | None = None,
    config_path: str | None = None,
) -> Dict:
    """Run minimal autotuning and export artifacts.

    Parameters
    ----------
    shape : tuple[int, int, int], optional
        Matrix shape (M, K, N). Overrides config if provided.
    trials : int, optional
        Number of trials. Overrides config if provided.
    work_dir : str, optional
        Working directory. Auto-generated if not provided.
    backend : str, optional
        Backend name (cpu, opencl, vulkan, hexagon). Overrides config if provided.
    export : str, optional
        Export format (so, tar). Overrides config if provided.
    config_path : str
        Path to TOML configuration file (required)

    Returns
    -------
    Dict
        {"work_dir": str, "artifacts": [str], "log": str}
    """
    if config_path is None:
        raise RuntimeError("config_path is required")
    
    # Load configuration with CLI overrides
    cli_overrides = {}
    if backend is not None:
        cli_overrides["backend"] = backend
    if trials is not None:
        cli_overrides["trials"] = trials
    if shape is not None:
        cli_overrides["shape"] = shape
    if export is not None:
        cli_overrides["export"] = export

    config = load_config(config_path, cli_overrides)

    # Determine working directory
    if work_dir is None:
        work_dir = os.path.join(
            config.log_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    os.makedirs(work_dir, exist_ok=True)

    # Capture log in-memory
    log_buf = io.StringIO()

    def _log(msg: str):
        print(msg)
        print(msg, file=log_buf)

    _log(f"[{_ts()}] Starting autotune with config: {config_path}")
    _log(f"[{_ts()}] Backend: {config.backend}")
    _log(f"[{_ts()}] Shapes: {config.shapes}")
    _log(f"[{_ts()}] Trials: {config.trials}")
    _log(f"[{_ts()}] Working directory: {work_dir}")

    # Set TVM_NDK_CC if specified in config
    if config.tvm_ndk_cc:
        os.environ["TVM_NDK_CC"] = config.tvm_ndk_cc

    # Set HEXAGON_TOOLCHAIN if specified in config
    if config.hexagon_toolchain:
        os.environ["HEXAGON_TOOLCHAIN"] = config.hexagon_toolchain

    # Validate export requirements
    if config.export == "so" and (not os.environ.get("TVM_NDK_CC") and not os.environ.get("HEXAGON_TOOLCHAIN")):
        raise RuntimeError(
            "TVM_NDK_CC or HEXAGON_TOOLCHAIN not set. For .so export, set TVM_NDK_CC to your Android NDK clang++ or HEXAGON_TOOLCHAIN to your Hexagon toolchain.\n"
            "Example: export TVM_NDK_CC=/path/to/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++\n"
            "Example: export HEXAGON_TOOLCHAIN=/path/to/hexagon-toolchain"
        )

    # Save effective config for reproducibility
    eff_config_path = os.path.join(work_dir, "config.toml")
    try:
        save_effective_config(config, config_path, eff_config_path)
        _log(f"[{_ts()}] Config saved -> {eff_config_path}")
    except Exception as e:
        _log(f"[{_ts()}] Config save failed: {e}")

    # Get targets
    _, remote_target = get_targets(config)
    _log(f"[{_ts()}] Remote target: {remote_target}")

    # Create builder and runner
    builder = create_builder(config, _log)
    runner = create_runner(config)

    # Process all shapes
    all_artifacts: List[str] = []
    for shape in config.shapes:
        _log(f"\n[{_ts()}] ===== Processing shape {shape} =====")
        artifacts = process_shape(
            shape=shape,
            config=config,
            remote_target=remote_target,
            work_dir=work_dir,
            builder=builder,
            runner=runner,
            logger=_log
        )
        all_artifacts.extend(artifacts)

    _log(f"\n[{_ts()}] ===== Autotune Complete =====")
    _log(f"[{_ts()}] Total artifacts: {len(all_artifacts)}")

    return {"work_dir": work_dir, "artifacts": all_artifacts, "log": log_buf.getvalue()}


def main():
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Minimal RPC autotune for TE matmul")
    p.add_argument("--trials", type=int, default=None, help="Number of tuning trials (overrides config)")
    p.add_argument("--backend", type=str, default=None, choices=["cpu", "opencl", "vulkan", "hexagon"], help="Backend (overrides config)")
    p.add_argument("--config", type=str, required=True, help="Path to autotune TOML config (required)")
    args = p.parse_args()

    res = run_autotune(trials=args.trials, backend=args.backend, config_path=args.config)
    print("\n=== Result ===")
    print(f"Work dir: {res['work_dir']}")
    for a in res["artifacts"]:
        print(f"Artifact: {a}")


if __name__ == "__main__":
    sys.exit(main())
