#!/usr/bin/env python3
"""Visualize tuning progress from TVM MetaSchedule log files.

This script parses TVM MetaSchedule log files and creates a visualization
showing how kernel performance (GFLOPS) improves over the number of device runs (trials).
"""

import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

try:
    import tomllib as _toml  # py311+
except Exception:
    try:
        import tomli as _toml  # py310-
    except Exception:
        _toml = None


def load_shapes_from_config(config_path: Path) -> List[Tuple[int, int, int]]:
    """Load shapes from TOML config file.
    
    Parameters
    ----------
    config_path : Path
        Path to the TOML config file
        
    Returns
    -------
    List[Tuple[int, int, int]]
        List of shapes (M, K, N) from config
    """
    if _toml is None:
        raise RuntimeError("Python tomllib/tomli not available; install tomli or use Python 3.11+")
    
    with open(config_path, "rb") as f:
        t = _toml.load(f)
    
    # Get shapes from autotune.common section
    autotune = t.get("autotune", {})
    common = autotune.get("common", {})
    cfg_shapes = common.get("shapes", [])
    
    shapes: List[Tuple[int, int, int]] = []
    for item in cfg_shapes:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            shapes.append((int(item[0]), int(item[1]), int(item[2])))
        elif isinstance(item, dict):
            shapes.append((int(item["m"]), int(item["k"]), int(item["n"])))
        else:
            raise ValueError(f"Invalid shape format: {item}")
    
    return shapes


def parse_log_file(
    log_path: Path,
    shapes: List[Tuple[int, int, int]]
) -> Dict[Tuple[int, int, int], Tuple[List[int], List[float], List[float]]]:
    """Parse TVM MetaSchedule log file to extract trial results grouped by shape.
    
    Parameters
    ----------
    log_path : Path
        Path to the log file
    shapes : List[Tuple[int, int, int]]
        List of shapes (M, K, N) in the order they appear in the log
        
    Returns
    -------
    shape_data : Dict[Tuple[int, int, int], Tuple[List[int], List[float], List[float]]]
        Dictionary mapping shape (M, K, N) to (trial_numbers, gflops, best_gflops)
    """
    shape_data = {shape: ([], [], []) for shape in shapes}
    
    # Pattern to match: [Task #0: main] Trial #N: GFLOPs: X.XXXX. Time: Y.YYYY us. Best GFLOPs: Z.ZZZZ
    trial_pattern = re.compile(
        r'\[Task #\d+: main\] Trial #(\d+): GFLOPs: ([\d.]+)\. Time: [\d.]+ us\. Best GFLOPs: ([\d.]+)'
    )
    
    # Pattern to match: Initializing Task #N: "main"
    init_pattern = re.compile(r'Initializing Task #\d+: "main"')
    
    lines = []
    with open(log_path, 'r') as f:
        lines = list(f)
    
    shape_index = 0
    current_shape = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a new task initialization (new shape)
        if init_pattern.search(line):
            # Use the next shape from the config list
            if shape_index < len(shapes):
                current_shape = shapes[shape_index]
                shape_index += 1
                print(f"Found new shape: {current_shape}")
            else:
                print(f"Warning: More shapes in log than in config. Using shape index {shape_index}")
                current_shape = None
        
        # Parse trial results
        match = trial_pattern.search(line)
        if match and current_shape is not None:
            trial_num = int(match.group(1))
            trial_gflops = float(match.group(2))
            best_gflops_so_far = float(match.group(3))
            
            trial_numbers, gflops, best_gflops = shape_data[current_shape]
            trial_numbers.append(trial_num)
            gflops.append(trial_gflops)
            best_gflops.append(best_gflops_so_far)
        
        i += 1
    
    # Remove shapes with no data
    return {shape: data for shape, data in shape_data.items() if data[0]}


def plot_tuning_progress(
    trial_numbers: List[int],
    gflops: List[float],
    best_gflops: List[float],
    shape: Tuple[int, int, int],
    output_path: Path = None,
    title: str = None
):
    """Create a visualization of tuning progress for a single shape.
    
    Parameters
    ----------
    trial_numbers : List[int]
        List of trial numbers
    gflops : List[float]
        List of GFLOPS for each trial
    best_gflops : List[float]
        List of best GFLOPS seen up to each trial
    shape : Tuple[int, int, int]
        Shape (M, K, N) for this plot
    output_path : Path, optional
        Path to save the plot. If None, displays interactively.
    title : str, optional
        Title for the plot. If None, uses a default title.
    """
    if not trial_numbers:
        print(f"No trial data found for shape {shape}!")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot individual trial GFLOPS (scatter plot, semi-transparent)
    ax.scatter(
        trial_numbers,
        gflops,
        alpha=0.3,
        s=20,
        color='blue',
        label='Individual Trials',
        zorder=1
    )
    
    # Plot best GFLOPS over time (line plot, more prominent)
    ax.plot(
        trial_numbers,
        best_gflops,
        color='red',
        linewidth=2,
        label='Best GFLOPS',
        zorder=2
    )
    
    # Add markers at key points
    if best_gflops:
        max_idx = best_gflops.index(max(best_gflops))
        ax.scatter(
            [trial_numbers[max_idx]],
            [best_gflops[max_idx]],
            color='green',
            s=200,
            marker='*',
            label=f'Peak: {best_gflops[max_idx]:.2f} GFLOPS',
            zorder=3,
            edgecolors='black',
            linewidths=1
        )
    
    ax.set_xlabel('Trial Number (Device Runs)', fontsize=12)
    ax.set_ylabel('GFLOPS', fontsize=12)
    m, k, n = shape
    shape_str = f"{m}x{k}x{n}"
    ax.set_title(
        title or f'Kernel Performance Improvement - Shape {shape_str}',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10)
    
    # Add statistics text box
    if trial_numbers and gflops:
        stats_text = (
            f'Shape: {shape_str}\n'
            f'Total Trials: {len(trial_numbers)}\n'
            f'Initial GFLOPS: {gflops[0]:.2f}\n'
            f'Final Best GFLOPS: {best_gflops[-1]:.2f}\n'
            f'Peak GFLOPS: {max(best_gflops):.2f}\n'
            f'Improvement: {((best_gflops[-1] / gflops[0]) - 1) * 100:.1f}%'
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_shapes(
    shape_data: Dict[Tuple[int, int, int], Tuple[List[int], List[float], List[float]]],
    output_dir: Path = None,
    base_title: str = None
):
    """Create separate plots for each shape.
    
    Parameters
    ----------
    shape_data : Dict[Tuple[int, int, int], Tuple[List[int], List[float], List[float]]]
        Dictionary mapping shape to trial data
    output_dir : Path, optional
        Directory to save plots. If None, displays interactively.
    base_title : str, optional
        Base title for plots
    """
    if not shape_data:
        print("No shape data found!")
        return
    
    print(f"Found {len(shape_data)} different shape(s)")
    
    for shape, (trial_numbers, gflops, best_gflops) in shape_data.items():
        m, k, n = shape
        shape_str = f"{m}x{k}x{n}"
        print(f"\nShape {shape_str}: {len(trial_numbers)} trials")
        print(f"  Initial GFLOPS: {gflops[0]:.2f}")
        print(f"  Final Best GFLOPS: {best_gflops[-1]:.2f}")
        print(f"  Peak GFLOPS: {max(best_gflops):.2f}")
        
        output_path = None
        if output_dir:
            output_path = output_dir / f"tuning_progress_{shape_str}.png"
        
        title = f"{base_title} - Shape {shape_str}" if base_title else None
        plot_tuning_progress(trial_numbers, gflops, best_gflops, shape, output_path, title)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize TVM MetaSchedule tuning progress from log files'
    )
    parser.add_argument(
        'log_file',
        type=str,
        help='Path to the TVM MetaSchedule log file (tvm.meta_schedule.logging.task_0_main.log)'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to the TOML config file (e.g., configs/opencl.toml) containing shape information'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for plots (e.g., ./plots). If not specified, displays interactively. Each shape will be saved as tuning_progress_{M}x{K}x{N}.png'
    )
    parser.add_argument(
        '-t', '--title',
        type=str,
        default=None,
        help='Base title for the plots'
    )
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return 1
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    print(f"Loading shapes from config: {config_path}")
    try:
        shapes = load_shapes_from_config(config_path)
        print(f"Found {len(shapes)} shape(s) in config: {shapes}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    print(f"Parsing log file: {log_path}")
    shape_data = parse_log_file(log_path, shapes)
    
    if not shape_data:
        print("Error: No trial data found in log file!")
        print("Make sure the log file contains lines with pattern:")
        print("  [Task #0: main] Trial #N: GFLOPs: X.XXXX. Time: Y.YYYY us. Best GFLOPs: Z.ZZZZ")
        return 1
    
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_all_shapes(shape_data, output_dir, args.title)
    
    return 0


if __name__ == '__main__':
    exit(main())

