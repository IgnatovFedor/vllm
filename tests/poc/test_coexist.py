"""Tests for PoC+Chat coexistence (chat-priority gating)."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.poc.routes import _generation_loop, _get_next_nonces


@pytest.fixture
def mock_engine_client():
    """Create a mock engine client for testing."""
    client = AsyncMock()
    client.poc_request = AsyncMock()
    client.poc_request.return_value = {
        "artifacts": [],
    }
    return client


class TestChatPriorityGating:
    """Tests for chat-priority gating in PoC GPU actions.

    Note: With the v1 architecture, coexistence checks are split:
    - AsyncLLM checks: processor.has_unfinished_requests(), dp_engines_running()
    - EngineCore checks: scheduler.has_requests()

    These tests verify the AsyncLLM-side checks. EngineCore checks are tested
    via integration tests or EngineCore unit tests.
    """

    @pytest.mark.asyncio
    async def test_generate_artifacts_skips_when_engine_step_in_progress(self):
        """Test generate_artifacts returns skip when engine step is in progress."""
        from vllm.v1.engine.async_llm import AsyncLLM

        # MPClient-like: dp_engines_running returns True
        engine_core = MagicMock()
        engine_core.dp_engines_running = MagicMock(return_value=True)

        output_processor = SimpleNamespace()
        output_processor.has_unfinished_requests = lambda: False

        async_llm = SimpleNamespace(
            engine_core=engine_core,
            output_processor=output_processor,
        )

        result = await AsyncLLM.poc_request(async_llm, "generate_artifacts", {
            "nonces": [0, 1, 2],
        })

        assert result["skipped"] is True
        assert result["reason"] == "engine_step_in_progress"
        # Should not call engine_core.poc_request_async
        engine_core.poc_request_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_artifacts_skips_when_chat_unfinished(self):
        """Test generate_artifacts returns skip when chat has unfinished requests."""
        from vllm.v1.engine.async_llm import AsyncLLM

        # No dp_engines_running (e.g. InprocClient)
        engine_core = MagicMock(spec=[])

        output_processor = SimpleNamespace()
        output_processor.has_unfinished_requests = lambda: True  # Chat unfinished

        async_llm = SimpleNamespace(
            engine_core=engine_core,
            output_processor=output_processor,
        )

        result = await AsyncLLM.poc_request(async_llm, "generate_artifacts", {
            "nonces": [0, 1, 2],
        })

        assert result["skipped"] is True
        assert result["reason"] == "chat_unfinished"

    @pytest.mark.asyncio
    async def test_generate_artifacts_proceeds_when_all_checks_pass(self):
        """Test generate_artifacts proceeds when AsyncLLM-side checks pass."""
        from vllm.v1.engine.async_llm import AsyncLLM

        # Mock engine_core that will return artifacts
        engine_core = MagicMock()
        engine_core.dp_engines_running = MagicMock(return_value=False)
        engine_core.poc_request_async = AsyncMock(return_value={
            "artifacts": [
                {"nonce": 0, "vector_b64": "AAA="},
                {"nonce": 1, "vector_b64": "BBB="},
            ]
        })

        output_processor = SimpleNamespace()
        output_processor.has_unfinished_requests = lambda: False

        async_llm = SimpleNamespace(
            engine_core=engine_core,
            output_processor=output_processor,
        )

        result = await AsyncLLM.poc_request(async_llm, "generate_artifacts", {
            "nonces": [0, 1],
            "block_hash": "hash",
            "public_key": "key",
            "seq_len": 256,
            "k_dim": 12,
        })

        engine_core.poc_request_async.assert_called_once()
        assert "skipped" not in result or result.get("skipped") is not True
        assert len(result["artifacts"]) == 2

    @pytest.mark.asyncio
    async def test_generate_artifacts_skips_when_scheduler_has_requests(self):
        """Test generate_artifacts returns skip when scheduler has pending requests.

        This tests the EngineCore-side check via the full async flow.
        """
        from vllm.v1.engine.async_llm import AsyncLLM

        # Mock engine_core that returns skip from EngineCore
        engine_core = MagicMock()
        engine_core.dp_engines_running = MagicMock(return_value=False)
        engine_core.poc_request_async = AsyncMock(return_value={
            "skipped": True,
            "reason": "pending_input",
            "artifacts": []
        })

        output_processor = SimpleNamespace()
        output_processor.has_unfinished_requests = lambda: False

        async_llm = SimpleNamespace(
            engine_core=engine_core,
            output_processor=output_processor,
        )

        result = await AsyncLLM.poc_request(async_llm, "generate_artifacts", {
            "nonces": [0, 1, 2],
        })

        assert result["skipped"] is True
        assert result["reason"] == "pending_input"


class TestGenerationLoopBackoff:
    """Tests for generation loop backoff behavior."""
    
    @pytest.mark.asyncio
    async def test_generation_loop_backs_off_on_skip(self, mock_engine_client):
        """Test that generation loop backs off when engine returns skipped."""
        stop_event = asyncio.Event()
        artifact_queue = asyncio.Queue()
        config = {
            "block_hash": "hash",
            "block_height": 100,
            "public_key": "key",
            "node_id": 0,
            "node_count": 1,
            "batch_size": 4,
            "seq_len": 256,
            "k_dim": 12,
        }
        stats = {"start_time": 0, "total_processed": 0}
        
        # Return skipped twice, then cancel
        call_count = 0
        async def mock_poc_request(action, payload, timeout_ms=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {"skipped": True, "artifacts": []}
            # Stop after 2 skips
            stop_event.set()
            return {"artifacts": [], "skipped": True}
        
        mock_engine_client.poc_request = mock_poc_request
        
        # Run the loop briefly
        with patch('vllm.poc.routes.POC_CHAT_BUSY_BACKOFF_SEC', 0.001):
            task = asyncio.create_task(
                _generation_loop(mock_engine_client, stop_event, artifact_queue, config, stats)
            )
            await asyncio.sleep(0.1)
            stop_event.set()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.CancelledError:
                pass
        
        # Should have called poc_request multiple times due to backoff retries
        assert call_count >= 2


class TestNonceGeneration:
    """Tests for API-side nonce generation."""
    
    def test_single_node_nonces(self):
        """Single node gets sequential nonces: 0, 1, 2, ..."""
        nonces, counter = _get_next_nonces(nonce_counter=0, batch_size=4, node_count=1)
        assert nonces == [0, 1, 2, 3]
        assert counter == 4
        
        nonces2, counter2 = _get_next_nonces(nonce_counter=counter, batch_size=4, node_count=1)
        assert nonces2 == [4, 5, 6, 7]
        assert counter2 == 8
    
    def test_multi_node_nonces_node0(self):
        """Node 0 of 3 gets: 0, 3, 6, 9, ..."""
        nonces, counter = _get_next_nonces(nonce_counter=0, batch_size=4, node_count=3)
        assert nonces == [0, 3, 6, 9]
        assert counter == 12
    
    def test_multi_node_nonces_node1(self):
        """Node 1 of 3 gets: 1, 4, 7, 10, ..."""
        nonces, counter = _get_next_nonces(nonce_counter=1, batch_size=4, node_count=3)
        assert nonces == [1, 4, 7, 10]
        assert counter == 13
    
    def test_multi_node_nonces_node2(self):
        """Node 2 of 3 gets: 2, 5, 8, 11, ..."""
        nonces, counter = _get_next_nonces(nonce_counter=2, batch_size=4, node_count=3)
        assert nonces == [2, 5, 8, 11]
        assert counter == 14


class TestUnknownAction:
    """Tests for unknown action handling."""

    @pytest.mark.asyncio
    async def test_v1_engine_rejects_unknown_action(self):
        """Test v1 EngineCore rejects unknown actions via AsyncLLM delegation."""
        from vllm.v1.engine.async_llm import AsyncLLM

        # Mock engine_core that raises ValueError for unknown action
        async def mock_poc_request_async(action, payload, timeout_ms=None):
            raise ValueError(f"Unknown PoC action: {action}")

        engine_core = MagicMock()
        engine_core.poc_request_async = mock_poc_request_async

        output_processor = SimpleNamespace()
        output_processor.has_unfinished_requests = lambda: False

        async_llm = SimpleNamespace(
            engine_core=engine_core,
            output_processor=output_processor,
        )

        with pytest.raises(ValueError, match="Unknown PoC action"):
            await AsyncLLM.poc_request(async_llm, "unknown_action", {})
