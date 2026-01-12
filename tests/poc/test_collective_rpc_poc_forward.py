"""Test execute_poc_forward via collective_rpc with real Qwen 0.6B model."""
import os
import pytest
from vllm import LLM
from vllm.poc.poc_model_runner import execute_poc_forward


def test_collective_rpc_execute_poc_forward(monkeypatch):
    """Test collective_rpc(execute_poc_forward, ...) with real model."""
    # Enable insecure serialization when multiprocessing is used
    # (needed for msgpack serialization of callable functions)
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Load Qwen 0.6B model
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        dtype="half",
        load_format="dummy",
        enforce_eager=True,
        gpu_memory_utilization=0.7,
        max_model_len=512,
    )

    # Get model_config and vllm_config
    model_config = llm.llm_engine.model_config
    vllm_config = llm.llm_engine.vllm_config

    # Parameters from tests and POC code
    block_hash = "test_hash"
    public_key = "test_key"
    nonces = [0, 1]
    seq_len = 4
    hidden_size = model_config.get_hidden_size()
    r_target = 1.5
    return_vectors = False

    # Call collective_rpc with execute_poc_forward (same as manager._run_forward)
    # Use llm_engine.collective_rpc directly instead of model_executor.collective_rpc
    results = llm.llm_engine.collective_rpc(
        execute_poc_forward,
        args=(
            block_hash,
            public_key,
            nonces,
            seq_len,
            hidden_size,
            r_target,
            vllm_config,
            return_vectors,
        ),
    )

    # Only the last PP rank returns a result (others return None)
    result = next((r for r in results if r is not None), None)

    # Verify result
    assert result is not None
    assert result["nonces"] == nonces
    assert len(result["distances"]) == len(nonces)
    assert all(0 <= d <= 2 for d in result["distances"]), "Distances should be in [0, 2]"
    assert "vectors" not in result, "vectors should not be returned when return_vectors=False"
