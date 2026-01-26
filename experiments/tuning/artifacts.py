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

"""Artifact building, exporting, and management."""

from __future__ import annotations
import os
import shutil
import tempfile
from typing import Optional, Callable, List, Tuple
import tvm
import tvm.tir
from tvm.contrib import ndk, hexagon
from tvm.contrib.hexagon.tools import export_module as hexagon_export_module

from config import AutotuneConfig, TargetVariant
from targets import get_variant_target


def export_module(mod: tvm.runtime.Module, out_dir: str, base_name: str, export_format: str = "so", backend: str = "android") -> str:
    """Export a TVM module to .so or .tar.
    
    Parameters
    ----------
    mod : tvm.runtime.Module
        Module to export
    out_dir : str
        Output directory
    base_name : str
        Base name for artifact (without extension)
    export_format : str
        Export format ("so" or "tar")
        
    Returns
    -------
    str
        Path to exported artifact
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if export_format == "so" and backend == "android" and os.environ.get("TVM_NDK_CC"):
        out_path = os.path.join(out_dir, base_name + ".so")
        tmp = os.path.join(tempfile.mkdtemp(), base_name + ".so")
        mod.export_library(tmp, fcompile=ndk.create_shared)
        shutil.copyfile(tmp, out_path)
        return out_path
    elif export_format == "so" and backend == "hexagon" and os.environ.get("HEXAGON_TOOLCHAIN"):
        out_path = os.path.join(out_dir, base_name + ".so")
        tmp = os.path.join(tempfile.mkdtemp(), base_name + ".so")
        mod.export_library(tmp, fcompile=hexagon.create_aot_shared, hexagon_arch="v75")
        shutil.copyfile(tmp, out_path)
        return out_path
    else:
        out_path = os.path.join(out_dir, base_name + ".tar")
        tmp = os.path.join(tempfile.mkdtemp(), base_name + ".tar")
        mod.export_library(tmp)
        shutil.copyfile(tmp, out_path)
        return out_path


def build_variant(
    sch_mod: tvm.IRModule,
    backend: str,
    base_target: tvm.target.Target,
    variant: TargetVariant,
    work_dir: str,
    base_name: str,
    export_format: str,
    logger: Callable[[str], None]
) -> Optional[str]:
    """Build a single target variant.
    
    Parameters
    ----------
    sch_mod : tvm.IRModule
        Scheduled module to build
    base_target : tvm.target.Target
        Base target
    variant : TargetVariant
        Variant configuration
    work_dir : str
        Output directory
    base_name : str
        Base artifact name (will be suffixed with variant name)
    export_format : str
        Export format ("so" or "tar")
    logger : Callable[[str], None]
        Logging function
        
    Returns
    -------
    Optional[str]
        Path to built artifact, or None if build failed
    """
    try:
        variant_target = get_variant_target(base_target, variant)
        
        logger(f"Building variant '{variant.name}': {variant_target}")
        rt_mod = tvm.build(sch_mod, target=variant_target)
        artifact_name = f"{base_name}_{variant.name}"
        art = export_module(rt_mod, work_dir, artifact_name, export_format=export_format, backend=backend)
        logger(f"Variant '{variant.name}' built successfully")
        return art
    except Exception as e:
        logger(f"Variant '{variant.name}' build failed: {e}")
        return None


def schedule_from_record(record) -> tvm.tir.Schedule:
    """Reconstruct schedule from tuning record.
    
    Parameters
    ----------
    record : TuningRecord
        Tuning record from MetaSchedule database
        
    Returns
    -------
    tvm.tir.Schedule
        Reconstructed schedule
    """
    sch = tvm.tir.Schedule(record.workload.mod)
    record.trace.apply_to_schedule(sch, remove_postproc=False)
    return sch


def save_trace(record, output_path: str, logger: Callable[[str], None]):
    """Save trace from tuning record to JSON file.
    
    Parameters
    ----------
    record : TuningRecord
        Tuning record
    output_path : str
        Output JSON file path
    logger : Callable[[str], None]
        Logging function
    """
    try:
        trace_txt = str(record.trace)
        with open(output_path, "w") as f:
            f.write(trace_txt)
        logger(f"Trace saved -> {output_path}")
    except Exception as e:
        logger(f"Trace save failed: {e}")


def build_all_variants(
    record,
    idx: int,
    shape_name: str,
    config: AutotuneConfig,
    base_target: tvm.target.Target,
    work_dir: str,
    logger: Callable[[str], None]
) -> List[Tuple[str, str, str]]:
    """Build all target variants for a tuning record.
    
    Parameters
    ----------
    record : TuningRecord
        Tuning record from database
    idx : int
        Record index (1-based)
    shape_name : str
        Shape identifier (e.g., "64x64x64")
    config : AutotuneConfig
        Configuration object
    base_target : tvm.target.Target
        Base target
    work_dir : str
        Output directory
    logger : Callable[[str], None]
        Logging function
        
    Returns
    -------
    List[Tuple[str, str, str]]
        List of (candidate_name, variant_name, artifact_path)
    """
    cand = f"cand{idx:03d}"
    artifacts: List[Tuple[str, str, str]] = []
    
    try:
        sch = schedule_from_record(record)
        sch_mod = sch.mod
        
        # Save trace
        trace_name = f"trace_{shape_name}_{cand}"
        trace_path = os.path.join(work_dir, f"{trace_name}.py")
        save_trace(record, trace_path, logger)
        
        # Build all variants
        for variant in config.target_variants:
            base_name = f"{shape_name}_{cand}"
            art = build_variant(
                sch_mod=sch_mod,
                backend=config.backend,
                base_target=base_target,
                variant=variant,
                work_dir=work_dir,
                base_name=base_name,
                export_format=config.export,
                logger=logger
            )
            if art:
                artifacts.append((cand, variant.name, art))
        
    except Exception as e:
        logger(f"Failed to build {cand}: {e}")
    
    return artifacts


def create_android_export_function(logger: Callable[[str], None]) -> Callable:
    """Create Android export function for LocalBuilder.
    
    Parameters
    ----------
    logger : Callable[[str], None]
        Logging function
        
    Returns
    -------
    Callable
        Export function for LocalBuilder
    """
    def _f_export(rt_mod: tvm.runtime.Module) -> str:
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "ms_tmp.so")
        logger(f"[Builder] Exporting module to {tmp_path}...")
        try:
            rt_mod.export_library(tmp_path, fcompile=ndk.create_shared)
            logger(f"[Builder] Module exported successfully")
        except Exception as e:
            logger(f"[Builder] Export failed: {e}")
            raise
        return tmp_path
    return _f_export

def create_hexagon_export_function(logger: Callable[[str], None]) -> Callable:
    """Create Hexagon export function for LocalBuilder.
    
    Parameters
    ----------
    logger : Callable[[str], None]
        Logging function
        
    Returns
    -------
    Callable
        Export function for LocalBuilder
    """
    def _f_export(rt_mod: tvm.runtime.Module) -> str:
        binary_path = hexagon_export_module(rt_mod, tempfile.mkdtemp())
        return str(binary_path)

    return _f_export