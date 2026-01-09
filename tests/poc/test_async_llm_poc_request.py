# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import ExitStack

import pytest

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine.async_llm import AsyncLLM

if not current_platform.is_cuda():
    pytest.skip(
        reason="V1 currently only supported on CUDA.", allow_module_level=True
    )


@pytest.fixture
def engine():
    """Shared engine fixture for PoC request tests."""
    with ExitStack() as after:
        with set_default_torch_num_threads(1):
            engine = AsyncLLM.from_engine_args(
                AsyncEngineArgs(model="Qwen/Qwen3-0.6B", enforce_eager=True, gpu_memory_utilization=0.7,
                                max_model_len=512)
            )
        after.callback(engine.shutdown)
        yield engine


@pytest.mark.asyncio
async def test_poc_request_init(engine):
    """Test poc_request with 'init' action."""
    payload = {
        "block_hash": "0x1234567890abcdef",
        "block_height": 100,
        "public_key": "test_public_key_123",
        "r_target": 0.5,
    }

    result = await engine.poc_request(action="init", payload=payload)

    assert result["status"] == "initialized"
    assert result["pow_status"]["state"] == "IDLE"
    assert result["pow_status"]["r_target"] == 0.5


@pytest.mark.asyncio
async def test_poc_request_status(engine):
    """Test poc_request("status") returns manager state."""
    # Initialize → check status → start_generate → check status
    payload = {
        "block_hash": "0x1234567890abcdef",
        "block_height": 100,
        "public_key": "test_public_key_123",
        "r_target": 0.5,
    }

    await engine.poc_request(action="init", payload=payload)
    status = await engine.poc_request(action="status", payload={})
    assert status["state"] == "IDLE"

    await engine.poc_request(action="start_generate", payload={})
    status = await engine.poc_request(action="status", payload={})
    assert status["state"] == "GENERATING"
