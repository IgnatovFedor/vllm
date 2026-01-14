#!/usr/bin/env python3
"""
PoC+Chat Coexistence E2E Test.

Tests that PoC continuous generation and /v1/chat/completions can coexist
on the same vLLM server with chat priority.

Usage:
    python scripts/poc_coexist_test.py
    python scripts/poc_coexist_test.py --model Qwen/Qwen3-0.6B

Requirements:
- Server must run in multiprocessing engine mode (default for OpenAI API server)
- Do NOT use --disable-frontend-multiprocessing
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

SERVER_PORT = 8766
SERVER_STARTUP_TIMEOUT = 120
POC_WARMUP_TIME = 5
CHAT_TIMEOUT = 60

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def start_vllm_server(model: str, log_file: Path) -> subprocess.Popen:
    """Start vLLM server with PoC enabled in MP mode."""
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    
    f = open(log_file, "w", buffering=1)
    proc = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--enable-poc",
            "--port", str(SERVER_PORT),
            "--gpu-memory-utilization", "0.4",
            "--max-model-len", "512",
            # NOTE: Do NOT add --disable-frontend-multiprocessing
            # MP mode is required for PoC+chat coexistence
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=Path.cwd(),
        start_new_session=True,
    )
    proc._log_file = f
    
    for i in range(SERVER_STARTUP_TIMEOUT):
        try:
            r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=1)
            if r.status_code == 200:
                return proc
        except:
            pass
        time.sleep(1)
        if i > 0 and i % 20 == 0:
            print(f"  Waiting for server... ({i}s)")
    
    proc.kill()
    f.close()
    raise RuntimeError(f"vLLM server failed to start within {SERVER_STARTUP_TIMEOUT}s")


def stop_process(proc: subprocess.Popen):
    """Stop a subprocess gracefully."""
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    if hasattr(proc, '_log_file') and proc._log_file:
        proc._log_file.flush()
        proc._log_file.close()


def api_call(method: str, endpoint: str, json_data: dict = None, timeout: int = 30) -> dict:
    """Make API call to vLLM server."""
    url = f"http://localhost:{SERVER_PORT}{endpoint}"
    if method == "GET":
        r = requests.get(url, timeout=timeout)
    else:
        r = requests.post(url, json=json_data, timeout=timeout)
    r.raise_for_status()
    return r.json()


def start_poc_generation(model: str) -> dict:
    """Start PoC continuous generation."""
    config = {
        "block_hash": "coexist_test_block",
        "block_height": 100,
        "public_key": "coexist_test_pubkey",
        "node_id": 0,
        "node_count": 1,
        "batch_size": 16,
        "params": {
            "model": model,
            "seq_len": 64,
            "k_dim": 12,
        },
    }
    return api_call("POST", "/api/v1/pow/init/generate", config)


def stop_poc_generation() -> dict:
    """Stop PoC generation."""
    return api_call("POST", "/api/v1/pow/stop")


def get_poc_status() -> dict:
    """Get PoC status."""
    return api_call("GET", "/api/v1/pow/status")


def send_chat_completion(model: str) -> dict:
    """Send a chat completion request."""
    request = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ],
        "max_tokens": 10,
        "temperature": 0.0,
    }
    return api_call("POST", "/v1/chat/completions", request, timeout=CHAT_TIMEOUT)


def main():
    parser = argparse.ArgumentParser(description="PoC+Chat Coexistence E2E Test")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model to test (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PoC+Chat Coexistence E2E Test")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Server port: {SERVER_PORT}")
    print()
    
    # Setup logs directory
    logs_dir = Path("logs/v2/coexist_test")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "server.log"
    
    server_proc = None
    results = {
        "server_started": False,
        "poc_started": False,
        "chat_completed_during_poc": False,
        "poc_resumed_after_chat": False,
    }
    
    try:
        # ====================================================================
        # Step 1: Start server
        # ====================================================================
        print("[1/5] Starting vLLM server (MP mode)...")
        server_proc = start_vllm_server(args.model, log_file)
        results["server_started"] = True
        print("      Server started")
        
        # ====================================================================
        # Step 2: Start PoC continuous generation
        # ====================================================================
        print("\n[2/5] Starting PoC continuous generation...")
        start_result = start_poc_generation(args.model)
        if start_result.get("status") == "OK":
            results["poc_started"] = True
            print("      PoC started")
        else:
            print(f"      FAILED: {start_result}")
            return 1
        
        # Let PoC run for a bit
        print(f"      Letting PoC warm up ({POC_WARMUP_TIME}s)...")
        time.sleep(POC_WARMUP_TIME)
        
        # Check PoC status
        status_before = get_poc_status()
        nonces_before = status_before.get("stats", {}).get("total_processed", 0)
        print(f"      PoC status: {status_before.get('status')}, processed: {nonces_before}")
        
        # ====================================================================
        # Step 3: Send chat completion while PoC is active
        # ====================================================================
        print("\n[3/5] Sending /v1/chat/completions while PoC is active...")
        try:
            chat_start = time.time()
            chat_result = send_chat_completion(args.model)
            chat_duration = time.time() - chat_start
            
            # Check if we got a valid response
            if chat_result.get("choices"):
                content = chat_result["choices"][0]["message"]["content"]
                results["chat_completed_during_poc"] = True
                print(f"      Chat completed in {chat_duration:.2f}s")
                print(f"      Response: {content[:50]}...")
            else:
                print(f"      FAILED: No choices in response")
                
        except requests.exceptions.Timeout:
            print(f"      FAILED: Chat request timed out after {CHAT_TIMEOUT}s")
        except Exception as e:
            print(f"      FAILED: {e}")
        
        # ====================================================================
        # Step 4: Check PoC resumed after chat
        # ====================================================================
        print("\n[4/5] Checking PoC resumed after chat...")
        time.sleep(3)  # Give PoC time to resume
        
        status_after = get_poc_status()
        nonces_after = status_after.get("stats", {}).get("total_processed", 0)
        
        if nonces_after > nonces_before:
            results["poc_resumed_after_chat"] = True
            print(f"      PoC resumed: {nonces_before} -> {nonces_after} nonces")
        else:
            print(f"      PoC did not progress: {nonces_before} -> {nonces_after}")
        
        # ====================================================================
        # Step 5: Stop PoC and report
        # ====================================================================
        print("\n[5/5] Stopping PoC...")
        stop_poc_generation()
        print("      PoC stopped")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if server_proc:
            stop_process(server_proc)
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    all_passed = True
    checks = [
        ("Server started", results["server_started"]),
        ("PoC started", results["poc_started"]),
        ("Chat completed during PoC", results["chat_completed_during_poc"]),
        ("PoC resumed after chat", results["poc_resumed_after_chat"]),
    ]
    
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CHECKS PASSED!")
        print(f"Logs saved to: {log_file}")
        return 0
    else:
        print("SOME CHECKS FAILED!")
        print(f"Check logs at: {log_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
