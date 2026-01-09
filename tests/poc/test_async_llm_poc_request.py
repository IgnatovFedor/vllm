from contextlib import ExitStack

import pytest

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine.async_llm import AsyncLLM

QWEN_ENGINE_ARGS = AsyncEngineArgs(
    model="Qwen/Qwen3-0.6B",
    enforce_eager=True,
    gpu_memory_utilization=0.7,
    max_model_len=512,
)


@pytest.mark.asyncio
async def test_poc_request_init():
    """Test poc_request with 'init' action."""
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(QWEN_ENGINE_ARGS)
        after.callback(engine.shutdown)

        # Prepare payload for PoCConfig
        payload = {
            "block_hash": "0x1234567890abcdef",
            "block_height": 100,
            "public_key": "test_public_key_123",
            "r_target": 0.5,
            "fraud_threshold": 0.01,
            "node_id": 0,
            "node_count": 1,
            "batch_size": 32,
            "seq_len": 256,
        }

        # Call poc_request with init action
        result = await engine.poc_request(action="init", payload=payload)

        # Verify the response structure
        assert "status" in result
        assert result["status"] == "initialized"
        assert "pow_status" in result

        # Verify that _poc_manager was created
        assert hasattr(engine, "_poc_manager")
        assert engine._poc_manager is not None

        # Verify the manager state
        pow_status = result["pow_status"]
        assert pow_status is not None
        assert "state" in pow_status
        assert pow_status["state"] == "IDLE"
        assert "valid_nonces" in pow_status
        assert "valid_distances" in pow_status
        assert "total_checked" in pow_status
        assert "total_valid" in pow_status
        assert "r_target" in pow_status
        assert pow_status["r_target"] == 0.5

