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

"""Target configuration and management."""

from __future__ import annotations
from typing import Tuple
import tvm
import tvm.target

from config import AutotuneConfig, TargetVariant


def get_targets(config: AutotuneConfig) -> Tuple[str, tvm.target.Target]:
    """Get local and remote targets based on config.
    
    Parameters
    ----------
    config : AutotuneConfig
        Configuration object
        
    Returns
    -------
    Tuple[str, tvm.target.Target]
        (local_target_str, remote_target_obj)
    """
    local_target = f"llvm -num-cores {config.num_cores}"
    # Host triple for GPU targets - add num-cores here
    # GPU devices don't have num-cores attribute directly, so we add it to the host
    host_target_str = f"llvm -mtriple=aarch64-linux-android -num-cores {config.num_cores}"
    host_triple = tvm.target.Target(host_target_str)
    
    if config.backend == "cpu":
        # Pure CPU on Android via llvm triple (include num-cores)
        remote_target = tvm.target.Target(
            f"llvm -mtriple=aarch64-linux-android -num-cores {config.num_cores}"
        )
    elif config.backend == "opencl":
        remote_target = _build_opencl_target(config, host_triple)
    elif config.backend == "vulkan":
        remote_target = _build_vulkan_target(config, host_triple)
    elif config.backend == "hexagon":
        remote_target = tvm.target.hexagon(cpu_ver="v75", hvx=128, use_qfloat=True, use_ieee_fp=True, num_cores=4)
        remote_target = remote_target.with_host(tvm.target.Target(
            f"llvm -mtriple=aarch64-linux-android -num-cores {config.num_cores}"
        ))

    else:
        raise RuntimeError(f"Unsupported backend: {config.backend}")
    
    return local_target, remote_target


def _build_opencl_target(config: AutotuneConfig, host_triple: tvm.target.Target) -> tvm.target.Target:
    """Build OpenCL target with device-specific parameters.
    
    Parameters
    ----------
    config : AutotuneConfig
        Configuration with device info
    host_triple : tvm.target.Target
        Host target (with num-cores)
        
    Returns
    -------
    tvm.target.Target
        Configured OpenCL target
    """
    # Start with base target
    target_str = "opencl -keys=opencl,gpu"
    
    # Get device info if available
    device_info = config.device_info or {}
    
    # Max threads per block
    max_threads = config.max_num_threads
    if max_threads is None and device_info:
        max_threads = device_info.get("max_work_group_size", 256)
    if max_threads:
        target_str += f" -max_num_threads={max_threads}"
        target_str += f" -max_threads_per_block={max_threads}"
    
    # Max shared memory per block (local memory)
    max_shared = config.max_shared_memory_per_block
    if max_shared is None and device_info:
        max_shared = device_info.get("local_mem_size", 32768)
    if max_shared:
        target_str += f" -max_shared_memory_per_block={max_shared}"
    
    # Max function args
    max_args = config.max_function_args or 128
    target_str += f" -max_function_args={max_args}"
    
    # Texture spatial limit
    texture_limit = config.texture_spatial_limit or 16384
    target_str += f" -texture_spatial_limit={texture_limit}"
    
    # Thread warp size
    warp_size = config.thread_warp_size or 32
    target_str += f" -thread_warp_size={warp_size}"
    
    return tvm.target.Target(target_str, host=host_triple)


def _build_vulkan_target(config: AutotuneConfig, host_triple: tvm.target.Target) -> tvm.target.Target:
    """Build Vulkan target with device-specific parameters.
    
    Parameters
    ----------
    config : AutotuneConfig
        Configuration with device info
    host_triple : tvm.target.Target
        Host target (with num-cores)
        
    Returns
    -------
    tvm.target.Target
        Configured Vulkan target
    """
    target_str = "vulkan"
    
    # Get device info if available
    device_info = config.device_info or {}
    
    # Max threads per block (similar to OpenCL work group size)
    max_threads = config.max_num_threads
    if max_threads is None and device_info:
        max_threads = device_info.get("max_work_group_size", 256)
    if max_threads:
        target_str += f" -max_num_threads={max_threads}"
        target_str += f" -max_threads_per_block={max_threads}"
    
    # Max shared memory per block
    max_shared = config.max_shared_memory_per_block
    if max_shared is None and device_info:
        max_shared = device_info.get("local_mem_size", 32768)
    if max_shared:
        target_str += f" -max_shared_memory_per_block={max_shared}"
    
    return tvm.target.Target(target_str, host=host_triple)


def _build_hexagon_target(config: AutotuneConfig, host_triple: tvm.target.Target) -> tvm.target.Target:
    target_str = "hexagon"
    local_target = f"llvm -num-cores {config.num_cores}"
    return tvm.target.Target(target_str, host=host_triple)


def get_variant_target(base_target: tvm.target.Target, variant: TargetVariant) -> tvm.target.Target:
    """Construct a target with variant-specific flags.
    
    Parameters
    ----------
    base_target : tvm.target.Target
        Base target object
    variant : TargetVariant
        Variant configuration
        
    Returns
    -------
    tvm.target.Target
        Target with variant flags applied
    """
    if isinstance(base_target, tvm.target.Target):
        # For GPU targets, return as-is (variants typically not used)
        if not variant.extra:
            return base_target
        # For CPU with extra flags, convert to string and append
        variant_target_str = f"{base_target} {variant.extra}".strip()
        return tvm.target.Target(variant_target_str)
    else:
        # For string targets, append variant flags
        variant_target_str = f"{base_target} {variant.extra}".strip()
        return tvm.target.Target(variant_target_str)


def verify_codegen_available(backend: str) -> Tuple[bool, str]:
    """Verify local codegen is available for the target.
    
    Parameters
    ----------
    backend : str
        Backend name (cpu, opencl, vulkan)
        
    Returns
    -------
    Tuple[bool, str]
        (is_available, error_message)
    """
    if backend == "vulkan":
        tgt_kind = "vulkan"
    elif backend == "opencl":
        tgt_kind = "opencl"
    else:
        tgt_kind = "llvm"
    
    build_sym = f"target.build.{tgt_kind}"
    build_fn = tvm.get_global_func(build_sym, allow_missing=True)
    
    if build_fn is None:
        if backend == "cpu":
            msg = (
                f"Local codegen for {tgt_kind} not found. "
                f"Ensure TVM was built with LLVM support."
            )
        else:
            msg = (
                f"Local codegen for {tgt_kind} not found. "
                f"Ensure TVM was built with USE_{backend.upper()}=ON."
            )
        return False, msg
    
    return True, ""
