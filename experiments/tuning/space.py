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

"""MetaSchedule space generation with device-specific optimizations."""

from __future__ import annotations
from typing import Optional, List
import tvm
from tvm import meta_schedule as ms

from config import AutotuneConfig


def create_schedule_space(config: AutotuneConfig):
    """Create MetaSchedule space generator based on backend and config.
    
    Parameters
    ----------
    config : AutotuneConfig
        Configuration object with backend and device info
        
    Returns
    -------
    SpaceGenerator or None
        Custom space generator, or None to use default
    """
    print(f"Creating schedule space for backend: {config.backend}")
    if config.backend == "opencl":
        return create_opencl_space(config)
    elif config.backend == "vulkan":
        return create_vulkan_space(config)
    elif config.backend == "hexagon":
        return create_hexagon_space(config)        
    elif config.backend == "cpu":
        return None  # Use default LLVM rules
    else:
        return None


def create_hexagon_space(config: AutotuneConfig):
    base_rules = ms.ScheduleRule.create("hexagon")
    postprocs = ms.Postproc.create("hexagon")
    mutator_probs = ms.Mutator.create("hexagon")
    
    space = ms.space_generator.PostOrderApply(
        sch_rules=base_rules,
        postprocs=postprocs,
        mutator_probs=mutator_probs,
    )
    
    return space


def create_opencl_space(config: AutotuneConfig):
    """Create OpenCL-specific schedule space.
    
    Uses device info to determine optimal vectorization width,
    unroll factors, and parallelization strategy.
    
    Parameters
    ----------
    config : AutotuneConfig
        Configuration with OpenCL device info
        
    Returns
    -------
    SpaceGenerator
        Configured space generator
    """
    # Get base GPU rules using ScheduleRule.create
    # "cuda" returns a list of default CUDA/GPU schedule rules
    base_rules = ms.ScheduleRule.create("cuda")
    
    # Extract device-specific parameters
    device_info = config.device_info or {}
    space_config = config.space_config or {}
    
    # Determine max vectorization width
    preferred_vec = device_info.get("preferred_vector_width_float", 1)
    if preferred_vec <= 0:
        preferred_vec = 1
    
    # Cap vectorization to avoid register pressure
    # Adreno typically prefers 2-4, but config can override
    max_vec = space_config.get("max_vectorize_extent", min(preferred_vec, 4))
    
    # Unroll steps to prevent code bloat
    unroll_steps = space_config.get("unroll_max_steps", [0, 16, 32, 64])
    
    # GPU targets don't have num-cores attribute, so disable max_jobs_per_core
    # This prevents MetaSchedule from trying to check num-cores
    max_jobs = -1
    
    # Explicit unrolling
    unroll_explicit = space_config.get("unroll_explicit", True)
    
    # Customize rules
    new_rules = []
    for rule in base_rules:
        if isinstance(rule, ms.schedule_rule.ParallelizeVectorizeUnroll):
            # Replace with device-optimized version
            new_rule = ms.schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=max_jobs,
                max_vectorize_extent=max_vec,
                unroll_max_steps=unroll_steps,
                unroll_explicit=unroll_explicit,
            )
            new_rules.append(new_rule)
        elif isinstance(rule, ms.schedule_rule.MultiLevelTiling):
            # Keep multi-level tiling but could customize tile sizes
            # based on device_info (max_work_group_size, local_mem_size, etc.)
            new_rules.append(rule)
        else:
            # Keep other rules as-is
            new_rules.append(rule)
    
    # Get default postprocessors and mutator probs from CUDA preset
    postprocs = ms.Postproc.create("cuda")
    mutator_probs = ms.Mutator.create("cuda")
    
    # Create space generator with customized rules
    space = ms.space_generator.PostOrderApply(
        sch_rules=new_rules,
        postprocs=postprocs,
        mutator_probs=mutator_probs,
    )
    
    return space


def create_vulkan_space(config: AutotuneConfig):
    """Create Vulkan-specific schedule space.
    
    Similar to OpenCL but with Vulkan-specific considerations.
    
    Parameters
    ----------
    config : AutotuneConfig
        Configuration with Vulkan device info
        
    Returns
    -------
    SpaceGenerator
        Configured space generator
    """
    # Vulkan compute shaders are similar to OpenCL
    # Use same strategy for now
    return create_opencl_space(config)


def get_schedule_rules_info(config: AutotuneConfig) -> str:
    """Get human-readable info about schedule rules being used.
    
    Parameters
    ----------
    config : AutotuneConfig
        Configuration object
        
    Returns
    -------
    str
        Description of schedule rules
    """
    device_info = config.device_info or {}
    space_config = config.space_config or {}
    
    if config.backend == "opencl":
        preferred_vec = device_info.get("preferred_vector_width_float", 1)
        max_vec = space_config.get("max_vectorize_extent", min(preferred_vec, 4))
        unroll_steps = space_config.get("unroll_max_steps", [0, 16, 32, 64])
        
        return f"""OpenCL Schedule Rules:
  - Max vectorize extent: {max_vec} (device preferred: {preferred_vec})
  - Unroll steps: {unroll_steps}
  - Max jobs per core: disabled (GPU target)
  - Device: {device_info.get('name', 'Unknown')}
  - Max work group size: {device_info.get('max_work_group_size', 'Unknown')}
  - Local memory: {device_info.get('local_mem_size', 'Unknown')} bytes"""
    
    elif config.backend == "vulkan":
        return f"""Vulkan Schedule Rules:
  - Using OpenCL-style rules
  - Device: {device_info.get('name', 'Unknown')}"""
    
    else:
        return f"{config.backend.upper()} Schedule Rules: Using defaults"
