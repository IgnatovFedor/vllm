#!/usr/bin/env python3
"""
Data collection script for PoC experiments (artifact-based protocol).

Collects nonces and artifacts (vector_b64) from multiple servers.

Config format:
{
    "model": "Qwen/Qwen3-0.6B",
    "seq_len": 256,
    "k_dim": 12,
    "block_hash": "...",  OR "block_hashes": ["...", "..."]
    "public_key": "...",  OR "public_keys": ["...", "..."]
    "block_height": 100,
    "batch_size": 128,
    "nonce_count": 500,
    "servers": {"name1": "http://...", "name2": "http://..."}
}

Usage:
    python scripts/collect_data.py --name my_experiment --config configs/servers.json
"""

import argparse
import base64
import itertools
import json
import os
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

# Global shutdown event for Ctrl-C handling
shutdown_event = threading.Event()
_sigint_count = 0
_sigint_lock = threading.Lock()


def signal_handler(signum, frame):
    """Handle Ctrl-C by initiating shutdown; force-exit on repeated Ctrl-C."""
    global _sigint_count
    with _sigint_lock:
        _sigint_count += 1
        count = _sigint_count

    shutdown_event.set()
    if count == 1:
        print("\n\nInterrupt received, shutting down... (press Ctrl-C again to force exit)")
    else:
        print("\n\nSecond interrupt received, forcing exit now.")
        os._exit(130)


def api_call(url: str, endpoint: str, method: str = "POST", json_data: dict = None) -> dict:
    """Make API call to server."""
    if shutdown_event.is_set():
        raise RuntimeError("Cancelled")
    full_url = f"{url}{endpoint}"
    if method == "GET":
        r = requests.get(full_url, timeout=30)
    else:
        r = requests.post(full_url, json=json_data, timeout=600)
    r.raise_for_status()
    return r.json()


def decode_vector(b64: str) -> np.ndarray:
    """Decode base64 FP16 little-endian to FP32."""
    data = base64.b64decode(b64)
    f16 = np.frombuffer(data, dtype='<f2')
    return f16.astype(np.float32)


def collect_from_server(name: str, url: str, config: dict, block_hash: str, public_key: str) -> dict:
    """Collect data from a single server for a specific seed."""
    # Stop any running generation
    try:
        api_call(url, "/api/v1/pow/stop")
    except Exception:
        pass  # Ignore if nothing running

    nonces = list(range(config.get("nonce_count", 500)))
    
    # Build generation request (new artifact-based API)
    gen_config = {
        "block_hash": block_hash,
        "block_height": config.get("block_height", 100),
        "public_key": public_key,
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {
            "model": config["model"],
            "seq_len": config.get("seq_len", 256),
            "k_dim": config.get("k_dim", 12),
        },
        "batch_size": config.get("batch_size", 128),
        "wait": True,
    }

    # Generate artifacts
    result = api_call(url, "/api/v1/pow/generate", json_data=gen_config)

    # Extract artifacts
    artifacts = result.get("artifacts", [])
    encoding = result.get("encoding", {"dtype": "f16", "k_dim": config.get("k_dim", 12), "endian": "le"})
    intermediates = result.get("intermediates")  # May be None (debug mode)
    
    # Decode vectors for analysis (store both base64 and decoded)
    decoded_vectors = []
    for artifact in artifacts:
        try:
            vec = decode_vector(artifact["vector_b64"])
            decoded_vectors.append(vec.tolist())
        except Exception:
            decoded_vectors.append(None)

    return {
        "server_name": name,
        "server_url": url,
        "block_hash": block_hash,
        "public_key": public_key,
        "nonces": [a["nonce"] for a in artifacts],
        "artifacts": artifacts,
        "vectors": decoded_vectors,
        "encoding": encoding,
        "intermediates": intermediates,  # Save for debugging
    }


def find_latest_run(name: str) -> Path | None:
    """Find the most recent output directory for given experiment name."""
    logs_dir = Path("logs/v2")
    if not logs_dir.exists():
        return None
    
    # Find directories matching pattern: {name}_{timestamp}
    matching = sorted(
        [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith(f"{name}_")],
        key=lambda d: d.name,
        reverse=True
    )
    return matching[0] if matching else None


def get_completed_tasks(out_dir: Path) -> set[str]:
    """Get set of completed task keys (filename stems) that have successful data."""
    completed = set()
    for json_file in out_dir.glob("*.json"):
        if json_file.name == "config.json":
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            # Check if it has actual data (not an error)
            if "error" not in data and data.get("artifacts"):
                completed.add(json_file.stem)
        except Exception:
            pass
    return completed


def get_output_filename(server_name: str, block_hash: str, public_key: str, multi_seed: bool) -> str:
    """Generate output filename based on seed mode."""
    if multi_seed:
        return f"{server_name}_{block_hash}_{public_key}.json"
    else:
        return f"{server_name}.json"


def get_task_key(server_name: str, block_hash: str, public_key: str, multi_seed: bool) -> str:
    """Generate task key for tracking completion."""
    if multi_seed:
        return f"{server_name}_{block_hash}_{public_key}"
    else:
        return server_name


def main():
    # Register signal handler for Ctrl-C
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="Collect PoC data from multiple servers")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--continue", dest="continue_run", action="store_true",
                        help="Continue from last run, skipping completed tasks")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = json.load(f)

    # Validate required config
    if "model" not in config:
        print("Error: config must include 'model' field")
        sys.exit(1)

    # Resolve seeds - support both single and multi-seed configs
    if "block_hashes" in config:
        block_hashes = config["block_hashes"]
    else:
        block_hashes = [config["block_hash"]]
    
    if "public_keys" in config:
        public_keys = config["public_keys"]
    else:
        public_keys = [config["public_key"]]
    
    seeds = list(itertools.product(block_hashes, public_keys))
    multi_seed = len(seeds) > 1

    # Determine output directory (under logs/v2/)
    if args.continue_run:
        out_dir = find_latest_run(args.name)
        if out_dir is None:
            print(f"No previous run found for '{args.name}', starting fresh")
            args.continue_run = False
    
    if not args.continue_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("logs/v2") / f"{args.name}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save config copy
        shutil.copy(config_path, out_dir / "config.json")
    
    # Get already completed tasks if continuing
    completed = get_completed_tasks(out_dir) if args.continue_run else set()

    # Build task list grouped by URL
    url_to_tasks = {}
    for name, url in config["servers"].items():
        for block_hash, public_key in seeds:
            task_key = get_task_key(name, block_hash, public_key, multi_seed)
            if task_key not in completed:
                url_to_tasks.setdefault(url, []).append((name, block_hash, public_key))

    total_tasks = sum(len(tasks) for tasks in url_to_tasks.values())
    
    print(f"Output: {out_dir}")
    print(f"Model: {config['model']}")
    print(f"Servers: {list(config['servers'].keys())}")
    print(f"Seeds: {len(seeds)} combinations")
    if multi_seed:
        print(f"  block_hashes: {block_hashes}")
        print(f"  public_keys: {public_keys}")
    print(f"Total tasks: {total_tasks}")
    print(f"Workers: {len(url_to_tasks)} (one per URL)")
    if completed:
        print(f"Skipping (already done): {len(completed)} tasks")
    print()

    def collect_all_seeds_for_url(url, task_list):
        """Process all seeds for one URL sequentially. Returns list of results."""
        results = []
        for name, block_hash, public_key in task_list:
            if shutdown_event.is_set():
                break
            
            task_key = get_task_key(name, block_hash, public_key, multi_seed)
            filename = get_output_filename(name, block_hash, public_key, multi_seed)
            
            try:
                result = collect_from_server(name, url, config, block_hash, public_key)
                with open(out_dir / filename, "w") as f:
                    json.dump(result, f, indent=2)
                results.append((name, block_hash, public_key, len(result["artifacts"]), None))
            except Exception as e:
                if shutdown_event.is_set():
                    break
                error_result = {
                    "server_name": name,
                    "server_url": url,
                    "block_hash": block_hash,
                    "public_key": public_key,
                    "error": str(e),
                }
                with open(out_dir / filename, "w") as f:
                    json.dump(error_result, f, indent=2)
                results.append((name, block_hash, public_key, 0, str(e)))
        
        return url, results

    if not url_to_tasks:
        print("No tasks to run.")
        return

    # Parallel execution - one worker per URL
    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=len(url_to_tasks)) as executor:
            futures = [
                executor.submit(collect_all_seeds_for_url, url, task_list)
                for url, task_list in url_to_tasks.items()
            ]

            pending = set(futures)
            while pending:
                if shutdown_event.is_set():
                    interrupted = True
                    for f in list(pending):
                        f.cancel()
                    break

                done_now = {f for f in pending if f.done()}
                if not done_now:
                    time.sleep(0.1)
                    continue

                for future in done_now:
                    pending.remove(future)
                    url, results = future.result()
                    for name, block_hash, public_key, artifact_count, error in results:
                        seed_str = f" [{block_hash}+{public_key}]" if multi_seed else ""
                        if error:
                            print(f"{name}{seed_str}: FAILED - {error}")
                        else:
                            print(f"{name}{seed_str}: OK ({artifact_count} artifacts)")
    except KeyboardInterrupt:
        interrupted = True
        shutdown_event.set()
        print("\n\nInterrupt received, cancelling pending tasks...")

    if interrupted:
        print(f"\nInterrupted. Partial results in {out_dir}")
        print("Use --continue to resume from where you left off.")
        sys.exit(1)
    else:
        print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
