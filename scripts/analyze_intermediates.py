#!/usr/bin/env python3
"""Compare intermediate states across different hardware to identify divergence points.

This script compares intermediate states from different servers (hardware) to find
where vector generation diverges. It can compare:
- Different servers from the same experiment
- Same or different servers from different experiments
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_server_data(exp_dir: Path) -> Dict[str, dict]:
    """Load all server data files from an experiment directory."""
    servers = {}
    for json_file in exp_dir.glob("*.json"):
        if json_file.name == "config.json":
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            server_name = data.get("server_name", json_file.stem)
            servers[server_name] = data
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    return servers


def compare_intermediates(
    data1: dict,
    data2: dict,
    server1_name: str,
    server2_name: str,
) -> Optional[Dict[str, dict]]:
    """Compare intermediate states between two server data files.
    
    Returns:
        Dict with comparison results for each intermediate stage, or None if no intermediates.
    """
    intermediates1 = data1.get("intermediates")
    intermediates2 = data2.get("intermediates")
    
    if intermediates1 is None:
        return None
    if intermediates2 is None:
        return None
    
    results = {}
    
    # Handle both dict and list formats
    if isinstance(intermediates1, list):
        if len(intermediates1) > 0 and isinstance(intermediates1[0], dict):
            intermediates1 = intermediates1[0]
        else:
            return None
    
    if isinstance(intermediates2, list):
        if len(intermediates2) > 0 and isinstance(intermediates2[0], dict):
            intermediates2 = intermediates2[0]
        else:
            return None
    
    # Compare each intermediate stage
    for key in sorted(set(intermediates1.keys()) | set(intermediates2.keys())):
        if key not in intermediates1:
            results[key] = {"error": "missing in server1"}
            continue
        if key not in intermediates2:
            results[key] = {"error": "missing in server2"}
            continue
        
        # Convert to numpy arrays
        arr1 = np.array(intermediates1[key])
        arr2 = np.array(intermediates2[key])
        
        if arr1.shape != arr2.shape:
            results[key] = {
                "error": f"shape mismatch {arr1.shape} vs {arr2.shape}"
            }
            continue
        
        # Check if arrays are identical
        if np.array_equal(arr1, arr2):
            results[key] = {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "max": 0.0,
                "exceed_counts": {thresh: 0 for thresh in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]},
                "worst_idx": 0,
                "worst_dist": 0.0,
                "worst_norm1": float(np.linalg.norm(arr1[0]) if len(arr1) > 0 else 0.0),
                "worst_norm2": float(np.linalg.norm(arr2[0]) if len(arr2) > 0 else 0.0),
                "note": "arrays are identical",
            }
            continue
        
        # Compute per-sample L2 distances
        diff = arr1 - arr2
        l2_dist = np.linalg.norm(diff, axis=-1)
        
        # Check if all distances are zero
        if np.all(l2_dist == 0.0):
            results[key] = {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "max": 0.0,
                "exceed_counts": {thresh: 0 for thresh in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]},
                "worst_idx": 0,
                "worst_dist": 0.0,
                "worst_norm1": float(np.linalg.norm(arr1[0]) if len(arr1) > 0 else 0.0),
                "worst_norm2": float(np.linalg.norm(arr2[0]) if len(arr2) > 0 else 0.0),
                "note": "all distances are exactly zero (suspicious - check if data is correct)",
            }
            continue
        
        # Statistics
        max_dist = float(np.max(l2_dist))
        mean_dist = float(np.mean(l2_dist))
        median_dist = float(np.median(l2_dist))
        std_dist = float(np.std(l2_dist))
        
        # Count how many exceed various thresholds
        thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
        exceed_counts = {}
        for thresh in thresholds:
            exceed_counts[thresh] = int(np.sum(l2_dist > thresh))
        
        # Find worst case
        worst_idx = int(np.argmax(l2_dist))
        worst_dist = float(l2_dist[worst_idx])
        
        results[key] = {
            "mean": mean_dist,
            "median": median_dist,
            "std": std_dist,
            "max": max_dist,
            "exceed_counts": exceed_counts,
            "worst_idx": worst_idx,
            "worst_dist": worst_dist,
            "worst_norm1": float(np.linalg.norm(arr1[worst_idx])),
            "worst_norm2": float(np.linalg.norm(arr2[worst_idx])),
        }
    
    return results


def print_comparison(
    server1_name: str,
    server2_name: str,
    results: Dict[str, dict],
):
    """Print comparison results in a readable format."""
    print(f"\n{'=' * 70}")
    print(f"Comparing: {server1_name} vs {server2_name}")
    print(f"{'=' * 70}")
    
    # Expected order of transformations
    expected_order = [
        "raw_last_hidden",
        "after_norm1",
        "after_pick",
        "after_haar",
        "after_norm2",
    ]
    
    # Print in expected order, then any others
    printed_keys = set()
    for key in expected_order:
        if key in results:
            printed_keys.add(key)
            _print_stage(key, results[key])
    
    # Print any remaining keys
    for key in sorted(results.keys()):
        if key not in printed_keys:
            _print_stage(key, results[key])


def _print_stage(key: str, result: dict):
    """Print statistics for a single intermediate stage."""
    print(f"\n  {key}:")
    
    if "error" in result:
        print(f"    ERROR: {result['error']}")
        return
    
    if "note" in result:
        print(f"    ⚠️  NOTE: {result['note']}")
        print(f"    mean={result['mean']:.6e}, median={result['median']:.6e}, "
              f"std={result['std']:.6e}, max={result['max']:.6e}")
        return
    
    print(f"    mean={result['mean']:.6e}, median={result['median']:.6e}, "
          f"std={result['std']:.6e}, max={result['max']:.6e}")
    
    # Show exceed counts for relevant thresholds
    exceed_str = ", ".join([
        f"{thresh:.0e}: {result['exceed_counts'][thresh]}"
        for thresh in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        if result['exceed_counts'][thresh] > 0
    ])
    if exceed_str:
        print(f"    exceed counts: {exceed_str}")
    
    # Show worst case if significant
    if result['worst_dist'] > 1e-6:
        print(f"    worst: idx={result['worst_idx']}, dist={result['worst_dist']:.6e}")
        print(f"      norm1={result['worst_norm1']:.6e}, norm2={result['worst_norm2']:.6e}")
    elif result['max'] > 0:
        print(f"    (all differences < 1e-6, but max={result['max']:.6e})")
        print(f"    sample1[0] norm={result['worst_norm1']:.6e}, sample2[0] norm={result['worst_norm2']:.6e}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_intermediates.py <dir1> [dir2] [--server1 NAME] [--server2 NAME]")
        print("")
        print("Compare intermediate states across different hardware.")
        print("")
        print("Arguments:")
        print("  dir1: Path to first experiment directory (logs/v2/...)")
        print("  dir2: (optional) Path to second experiment directory")
        print("        If omitted, compares all servers within dir1")
        print("")
        print("Options:")
        print("  --server1 NAME: Compare specific server from dir1")
        print("  --server2 NAME: Compare specific server from dir2 (or dir1 if dir2 omitted)")
        print("")
        print("Examples:")
        print("  # Compare all server pairs in one experiment")
        print("  python analyze_intermediates.py logs/v2/my_experiment")
        print("")
        print("  # Compare V100 vs 4070s from same experiment")
        print("  python analyze_intermediates.py logs/v2/my_experiment --server1 V100 --server2 4070s")
        print("")
        print("  # Compare V100 from exp1 vs 4070s from exp2")
        print("  python analyze_intermediates.py logs/v2/exp1 logs/v2/exp2 --server1 V100 --server2 4070s")
        sys.exit(1)
    
    # Parse arguments
    dir1 = Path(sys.argv[1])
    dir2 = None
    server1_name = None
    server2_name = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--server1" and i + 1 < len(sys.argv):
            server1_name = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--server2" and i + 1 < len(sys.argv):
            server2_name = sys.argv[i + 1]
            i += 2
        elif dir2 is None and not sys.argv[i].startswith("--"):
            dir2 = Path(sys.argv[i])
            i += 1
        else:
            i += 1
    
    if not dir1.exists():
        print(f"Error: {dir1} does not exist")
        sys.exit(1)
    
    # Load server data
    servers1 = load_server_data(dir1)
    if not servers1:
        print(f"Error: No server data found in {dir1}")
        sys.exit(1)
    
    if dir2:
        if not dir2.exists():
            print(f"Error: {dir2} does not exist")
            sys.exit(1)
        servers2 = load_server_data(dir2)
        if not servers2:
            print(f"Error: No server data found in {dir2}")
            sys.exit(1)
    else:
        servers2 = servers1  # Compare within same experiment
    
    # Determine which comparisons to make
    comparisons: List[Tuple[str, str, dict, dict]] = []
    
    if server1_name and server2_name:
        # Specific comparison
        if server1_name not in servers1:
            print(f"Error: Server '{server1_name}' not found in {dir1}")
            sys.exit(1)
        if server2_name not in servers2:
            print(f"Error: Server '{server2_name}' not found in {dir2 or dir1}")
            sys.exit(1)
        comparisons.append((server1_name, server2_name, servers1[server1_name], servers2[server2_name]))
    elif server1_name:
        # Compare server1 with all servers in dir2 (or dir1)
        if server1_name not in servers1:
            print(f"Error: Server '{server1_name}' not found in {dir1}")
            sys.exit(1)
        for name2, data2 in servers2.items():
            if name2 != server1_name:  # Skip self-comparison
                comparisons.append((server1_name, name2, servers1[server1_name], data2))
    else:
        # Compare all pairs
        for name1, data1 in servers1.items():
            for name2, data2 in servers2.items():
                if dir2 or name1 < name2:  # If same dir, only compare each pair once
                    comparisons.append((name1, name2, data1, data2))
    
    if not comparisons:
        print("No comparisons to make")
        sys.exit(0)
    
    # Run comparisons
    for server1_name, server2_name, data1, data2 in comparisons:
        results = compare_intermediates(data1, data2, server1_name, server2_name)
        
        if results is None:
            print(f"\n{server1_name} vs {server2_name}: No intermediates found")
            continue
        
        print_comparison(server1_name, server2_name, results)


if __name__ == "__main__":
    main()

