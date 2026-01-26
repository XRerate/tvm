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

"""Configuration management for autotuning."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

try:
    import tomllib as _toml  # py311+
except Exception:
    try:
        import tomli as _toml  # py310-
    except Exception:
        _toml = None


@dataclass
class TargetVariant:
    """Configuration for a target build variant."""
    name: str
    extra: str = ""


@dataclass
class AutotuneConfig:
    """Complete autotuning configuration."""
    # Backend settings
    backend: str
    
    # RPC tracker settings
    tracker_host: str
    tracker_port: int
    tracker_key: str
    bench_key: str
    
    # Search strategy
    strategy: str = "evolutionary"
    database: str = "json"
    seed: Optional[int] = 42  # Random seed for reproducibility
    
    # Build settings
    tvm_ndk_cc: Optional[str] = None
    hexagon_toolchain: Optional[str] = None
    export: str = "so"
    
    # Tuning settings
    trials: int = 3
    num_trials_per_iter: int = 1
    shapes: List[Tuple[int, int, int]] = field(default_factory=lambda: [(64, 64, 64)])
    
    # Target settings
    num_cores: int = 40
    max_shared_memory_per_block: Optional[int] = None
    target_variants: List[TargetVariant] = field(default_factory=lambda: [TargetVariant("base", "")])
    
    # OpenCL/Vulkan device info (optional)
    device_info: Optional[Dict] = None
    
    # Advanced target config (optional overrides)
    max_num_threads: Optional[int] = None
    max_function_args: Optional[int] = None
    texture_spatial_limit: Optional[int] = None
    thread_warp_size: Optional[int] = None
    
    # MetaSchedule space config (optional)
    space_config: Optional[Dict] = None
    
    # Directory settings
    log_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "tuning_logs"))
    
    # Timeout settings
    builder_timeout_sec: float = 180.0
    session_timeout_sec: int = 120
    session_priority: int = 1
    
    # Benchmark settings
    bench_session_timeout_sec: int = 60
    bench_oom_retries: int = 2
    bench_oom_cooldown_sec: int = 5
    
    # Evaluator settings
    eval_number: int = 1
    eval_repeat: int = 1
    eval_min_repeat_ms: int = 0
    eval_enable_cpu_cache_flush: bool = False

    def validate(self):
        """Validate configuration values."""
        if self.backend not in ("cpu", "opencl", "vulkan", "hexagon"):
            raise ValueError(f"Invalid backend: {self.backend}")
        if self.strategy not in ("evolutionary", "nsgaii"):
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be 'evolutionary' or 'nsgaii'")
        if self.database not in ("json", "json_pareto"):
            raise ValueError(f"Invalid database: {self.database}. Must be 'json' or 'json_pareto'")
        if self.export not in ("so", "tar"):
            raise ValueError(f"Invalid export format: {self.export}")
        if self.trials < 1:
            raise ValueError(f"trials must be >= 1, got {self.trials}")
        if not self.shapes:
            raise ValueError("At least one shape must be specified")
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        result = {
            "backend": self.backend,
            "strategy": self.strategy,
            "database": self.database,
            "seed": self.seed,
            "tracker_host": self.tracker_host,
            "tracker_port": self.tracker_port,
            "tracker_key": self.tracker_key,
            "bench_key": self.bench_key,
            "tvm_ndk_cc": self.tvm_ndk_cc,
            "hexagon_toolchain": self.hexagon_toolchain,
            "export": self.export,
            "trials": self.trials,
            "num_trials_per_iter": self.num_trials_per_iter,
            "shapes": self.shapes,
            "num_cores": self.num_cores,
            "max_shared_memory_per_block": self.max_shared_memory_per_block,
            "target_variants": [{"name": v.name, "extra": v.extra} for v in self.target_variants],
            "log_dir": self.log_dir,
            "builder_timeout_sec": self.builder_timeout_sec,
            "session_timeout_sec": self.session_timeout_sec,
            "session_priority": self.session_priority,
            "bench_session_timeout_sec": self.bench_session_timeout_sec,
            "bench_oom_retries": self.bench_oom_retries,
            "bench_oom_cooldown_sec": self.bench_oom_cooldown_sec,
            "eval_number": self.eval_number,
            "eval_repeat": self.eval_repeat,
            "eval_min_repeat_ms": self.eval_min_repeat_ms,
            "eval_enable_cpu_cache_flush": self.eval_enable_cpu_cache_flush,
        }
        
        # Add device_info if present
        if self.device_info:
            result["device_info"] = self.device_info
        
        # Add advanced target config if present
        if self.max_num_threads:
            result["max_num_threads"] = self.max_num_threads
        if self.max_function_args:
            result["max_function_args"] = self.max_function_args
        if self.texture_spatial_limit:
            result["texture_spatial_limit"] = self.texture_spatial_limit
        if self.thread_warp_size:
            result["thread_warp_size"] = self.thread_warp_size
            
        return result


def _parse_shapes(cfg_shapes) -> List[Tuple[int, int, int]]:
    """Parse shapes from config."""
    shapes: List[Tuple[int, int, int]] = []
    for item in cfg_shapes:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            shapes.append((int(item[0]), int(item[1]), int(item[2])))
        elif isinstance(item, dict):
            shapes.append((int(item["m"]), int(item["k"]), int(item["n"])))
        else:
            raise ValueError(f"Invalid shape format: {item}")
    return shapes


def _parse_target_variants(cfg_variants) -> List[TargetVariant]:
    """Parse target variants from config."""
    variants: List[TargetVariant] = []
    for v in cfg_variants:
        if isinstance(v, dict) and "name" in v:
            variants.append(TargetVariant(name=str(v["name"]), extra=str(v.get("extra", ""))))
        else:
            raise ValueError(f"Invalid target_variant format: {v}")
    return variants if variants else [TargetVariant("base", "")]


def load_config(config_path: str, cli_overrides: Optional[Dict] = None) -> AutotuneConfig:
    """Load configuration from TOML file with optional CLI overrides.
    
    Parameters
    ----------
    config_path : str
        Path to TOML configuration file
    cli_overrides : Optional[Dict]
        CLI overrides for config values (backend, trials, shape, export, etc.)
        
    Returns
    -------
    AutotuneConfig
        Validated configuration object
    """
    if _toml is None:
        raise RuntimeError("Python tomllib/tomli not available; install tomli or use Python 3.11+")
    
    with open(config_path, "rb") as f:
        t = _toml.load(f)
    
    # Get autotune section
    autotune = t.get("autotune", {})
    if not autotune:
        raise RuntimeError("Config must have [autotune] section with subsections like [autotune.common]")
    
    # Start with common settings
    cfg = dict(autotune.get("common", {}))
    
    # Determine backend from common or backend-specific section
    backend = cfg.get("backend")
    if not backend:
        # Try to infer from available backend sections
        for possible_backend in ["cpu", "opencl", "vulkan", "hexagon"]:
            if possible_backend in autotune:
                backend = possible_backend
                break
    
    if not backend:
        raise RuntimeError("backend must be specified in config")
    
    # Merge backend-specific settings
    if backend in autotune:
        backend_cfg = autotune[backend]
        cfg.update(backend_cfg)
    
    # Apply CLI overrides
    if cli_overrides:
        if cli_overrides.get("backend"):
            backend = cli_overrides["backend"]
            cfg["backend"] = backend
            # Re-merge backend-specific settings
            if backend in autotune:
                cfg.update(autotune[backend])
        
        if cli_overrides.get("strategy"):
            cfg["strategy"] = cli_overrides["strategy"]
        
        if cli_overrides.get("database"):
            cfg["database"] = cli_overrides["database"]
        
        if cli_overrides.get("seed") is not None:
            cfg["seed"] = cli_overrides["seed"]
        
        if cli_overrides.get("trials") and cli_overrides["trials"] != 3:  # 3 is default
            cfg["trials"] = cli_overrides["trials"]
        
        if cli_overrides.get("shape") and cli_overrides["shape"] != (64, 64, 64):  # default
            cfg["shapes"] = [cli_overrides["shape"]]
        
        if cli_overrides.get("export"):
            cfg["export"] = cli_overrides["export"]
    
    # Validate required fields
    required = ["tracker_host", "tracker_port", "tracker_key"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise RuntimeError(f"Missing required config fields: {', '.join(missing)}")
    
    # Parse shapes
    cfg_shapes = cfg.get("shapes", [[64, 64, 64]])
    shapes = _parse_shapes(cfg_shapes)
    
    # Parse target variants
    cfg_variants = cfg.get("target_variants", [])
    target_variants = _parse_target_variants(cfg_variants) if cfg_variants else [TargetVariant("base", "")]
    
    # Parse device info (for GPU backends)
    device_info = None
    if f"{backend}.device_info" in cfg or "device_info" in cfg:
        device_info = dict(cfg.get("device_info", {}))
    
    # Parse space config (for MetaSchedule)
    space_config = None
    if f"{backend}.space" in cfg or "space" in cfg:
        space_config = dict(cfg.get("space", {}))
    
    # Build config object
    config = AutotuneConfig(
        backend=str(cfg["backend"]).lower(),
        tracker_host=str(cfg["tracker_host"]),
        tracker_port=int(cfg["tracker_port"]),
        tracker_key=str(cfg["tracker_key"]),
        bench_key=str(cfg.get("bench_key", cfg["tracker_key"])),
        strategy=str(cfg.get("strategy", "evolutionary")),
        database=str(cfg.get("database", "json")),
        seed=int(cfg["seed"]) if "seed" in cfg else 42,
        tvm_ndk_cc=cfg.get("tvm_ndk_cc"),
        hexagon_toolchain=cfg.get("hexagon_toolchain"),
        export=str(cfg.get("export", "so")).lower(),
        trials=int(cfg.get("trials", 3)),
        num_trials_per_iter=int(cfg.get("num_trials_per_iter", 1)),
        shapes=shapes,
        num_cores=int(cfg.get("num_cores", 40)),
        max_shared_memory_per_block=int(cfg["max_shared_memory_per_block"]) if "max_shared_memory_per_block" in cfg else None,
        target_variants=target_variants,
        device_info=device_info,
        max_num_threads=int(cfg["max_num_threads"]) if "max_num_threads" in cfg else None,
        max_function_args=int(cfg["max_function_args"]) if "max_function_args" in cfg else None,
        texture_spatial_limit=int(cfg["texture_spatial_limit"]) if "texture_spatial_limit" in cfg else None,
        thread_warp_size=int(cfg["thread_warp_size"]) if "thread_warp_size" in cfg else None,
        space_config=space_config,
        log_dir=str(cfg.get("log_dir", os.path.join(os.getcwd(), "tuning_logs"))),
        builder_timeout_sec=float(cfg.get("builder_timeout_sec", 180.0)),
        session_timeout_sec=int(cfg.get("session_timeout_sec", 120)),
        session_priority=int(cfg.get("session_priority", 1)),
        bench_session_timeout_sec=int(cfg.get("bench_session_timeout_sec", 60)),
        bench_oom_retries=int(cfg.get("bench_oom_retries", 2)),
        bench_oom_cooldown_sec=int(cfg.get("bench_oom_cooldown_sec", 5)),
        eval_number=int(cfg.get("eval_number", 1)),
        eval_repeat=int(cfg.get("eval_repeat", 1)),
        eval_min_repeat_ms=int(cfg.get("eval_min_repeat_ms", 0)),
        eval_enable_cpu_cache_flush=bool(cfg.get("eval_enable_cpu_cache_flush", False)),
    )
    
    config.validate()
    return config


def save_effective_config(config: AutotuneConfig, config_path: str, output_path: str):
    """Save effective configuration for reproducibility.
    
    Parameters
    ----------
    config : AutotuneConfig
        Configuration to save
    config_path : str
        Original config file path (for reference)
    output_path : str
        Output TOML file path
    """
    with open(output_path, "w") as f:
        f.write("# Auto-generated effective config\n")
        f.write(f"# Source: {os.path.abspath(config_path)}\n")
        f.write("# Feed back with --config for exact reproduction\n\n")
        f.write("[autotune]\n")
        f.write(f'backend = "{config.backend}"\n')
        f.write(f'strategy = "{config.strategy}"\n')
        f.write(f'database = "{config.database}"\n')
        f.write(f'seed = {config.seed}\n')
        f.write(f'tracker_host = "{config.tracker_host}"\n')
        f.write(f'tracker_port = {config.tracker_port}\n')
        f.write(f'tracker_key = "{config.tracker_key}"\n')
        f.write(f'bench_key = "{config.bench_key}"\n')
        if config.tvm_ndk_cc:
            f.write(f'tvm_ndk_cc = "{config.tvm_ndk_cc}"\n')
        if config.hexagon_toolchain:
            f.write(f'hexagon_toolchain = "{config.hexagon_toolchain}"\n')
        f.write(f'export = "{config.export}"\n')
        f.write(f'trials = {config.trials}\n')
        f.write(f'num_trials_per_iter = {config.num_trials_per_iter}\n')
        f.write(f'num_cores = {config.num_cores}\n')
        f.write(f'log_dir = "{config.log_dir}"\n')
        if config.max_shared_memory_per_block is not None:
            f.write(f'max_shared_memory_per_block = {config.max_shared_memory_per_block}\n')
        
        # Timeout settings
        f.write(f'builder_timeout_sec = {config.builder_timeout_sec}\n')
        f.write(f'session_timeout_sec = {config.session_timeout_sec}\n')
        f.write(f'session_priority = {config.session_priority}\n')
        
        # Benchmark settings
        f.write(f'bench_session_timeout_sec = {config.bench_session_timeout_sec}\n')
        f.write(f'bench_oom_retries = {config.bench_oom_retries}\n')
        f.write(f'bench_oom_cooldown_sec = {config.bench_oom_cooldown_sec}\n')
        
        # Evaluator settings
        f.write(f'eval_number = {config.eval_number}\n')
        f.write(f'eval_repeat = {config.eval_repeat}\n')
        f.write(f'eval_min_repeat_ms = {config.eval_min_repeat_ms}\n')
        f.write(f'eval_enable_cpu_cache_flush = {str(config.eval_enable_cpu_cache_flush).lower()}\n')
        
        f.write("\nshapes = [\n")
        for (m, k, n) in config.shapes:
            f.write(f'  [{m}, {k}, {n}],\n')
        f.write("]\n")
        
        # Write target variants
        if len(config.target_variants) > 1 or config.target_variants[0].name != "base" or config.target_variants[0].extra:
            f.write("\ntarget_variants = [\n")
            for v in config.target_variants:
                if v.extra:
                    f.write(f'  {{name = "{v.name}", extra = "{v.extra}"}},\n')
                else:
                    f.write(f'  {{name = "{v.name}"}},\n')
            f.write("]\n")
