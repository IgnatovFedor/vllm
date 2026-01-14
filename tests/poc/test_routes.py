"""Tests for PoC API routes (artifact-based protocol with API-owned state)."""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.poc.routes import (
    router, _poc_tasks, _is_generation_active, _get_next_nonces,
    _generate_queue, _generate_results, _generate_worker_task,
    _ensure_queue_initialized, _cleanup_old_results, GenerateResult,
    GENERATE_RESULT_TTL_SEC,
)
from vllm.poc.config import PoCState


# Mock generation loop that waits on stop_event (simulates real loop)
async def _mock_generation_loop(engine_client, stop_event, artifact_queue, config, stats):
    """Mock generation loop that waits until stop_event is set."""
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass


@pytest.fixture
def mock_engine_client():
    """Create a mock engine client for testing."""
    client = AsyncMock()
    
    # Default: generate_artifacts returns empty (no artifacts)
    client.poc_request.return_value = {
        "artifacts": [],
    }
    
    return client


@pytest.fixture
def app_with_poc(mock_engine_client):
    """Create a FastAPI app with PoC router and mocked engine."""
    app = FastAPI()
    app.include_router(router)
    
    # Set up app state
    app.state.engine_client = mock_engine_client
    app.state.poc_enabled = True
    app.state.poc_deployed = {
        "model": "test-model",
        "seq_len": 256,
        "k_dim": 12,
    }
    
    # Set up mock openai_serving_models for params checking
    mock_base_path = MagicMock()
    mock_base_path.model_path = "test-model"
    mock_base_path.name = "test-model"
    
    mock_serving_models = MagicMock()
    mock_serving_models.base_model_paths = [mock_base_path]
    app.state.openai_serving_models = mock_serving_models
    
    return app


@pytest.fixture
def client(app_with_poc):
    """Create test client with mocked generation loop."""
    _poc_tasks.clear()
    # Patch the generation loop to avoid infinite loops in tests
    with patch('vllm.poc.routes._generation_loop', _mock_generation_loop):
        yield TestClient(app_with_poc)
    # Clean up
    _cancel_all_poc_tasks()


def _cancel_all_poc_tasks():
    """Cancel all running PoC tasks synchronously."""
    for app_id, tasks in list(_poc_tasks.items()):
        if tasks.get("stop_event"):
            tasks["stop_event"].set()
        if tasks.get("gen_task"):
            tasks["gen_task"].cancel()
        if tasks.get("send_task"):
            tasks["send_task"].cancel()
    _poc_tasks.clear()


class TestPoCInitGenerate:
    def test_init_generate_starts_generation(self, client, mock_engine_client):
        """Test /init/generate starts background generation (API-owned state)."""
        # No engine status/init/start_generate calls needed anymore
        # Just need generate_artifacts to work
        mock_engine_client.poc_request.return_value = {
            "artifacts": [
                {"nonce": 0, "vector_b64": "AAAA"},
            ],
        }
        
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "batch_size": 32,
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"
        assert data["pow_status"]["status"] == "GENERATING"

    def test_init_generate_conflict_when_already_generating(self, client, app_with_poc, mock_engine_client):
        """Test /init/generate returns 409 when already generating (API-owned conflict)."""
        app_id = id(app_with_poc)
        
        # Simulate active generation by creating a mock task that's not done
        mock_task = MagicMock()
        mock_task.done.return_value = False
        _poc_tasks[app_id] = {
            "gen_task": mock_task,
            "stop_event": asyncio.Event(),
            "config": {"block_hash": "existing"},
            "stats": {},
        }
        
        # Second call should conflict (API-owned state)
        response2 = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc456",
            "block_height": 101,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
        })
        
        assert response2.status_code == 409
        assert "Already generating" in response2.json()["detail"]

    def test_init_generate_params_mismatch(self, client, mock_engine_client):
        """Test /init/generate returns 409 when params don't match deployed."""
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "params": {
                "model": "wrong-model",  # Doesn't match deployed
                "seq_len": 256,
                "k_dim": 12,
            },
        })
        
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "model mismatch" in detail.lower()
        assert "wrong-model" in detail


class TestPoCGenerate:
    def test_generate_returns_artifacts(self, client, mock_engine_client):
        """Test /generate computes artifacts for given nonces."""
        # Only generate_artifacts is called now (no status check)
        mock_engine_client.poc_request.return_value = {
            "artifacts": [
                {"nonce": 0, "vector_b64": "AAAAAAA="},
                {"nonce": 1, "vector_b64": "BBBBBBB="},
            ],
        }
        
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "nonces": [0, 1],
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
            "wait": True,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert len(data["artifacts"]) == 2
        assert data["encoding"]["dtype"] == "f16"

    def test_generate_queues_when_init_generate_active(self, client, app_with_poc, mock_engine_client):
        """Test /generate with wait=false queues when /init/generate is running."""
        app_id = id(app_with_poc)
        
        # Simulate active generation by creating a mock task that's not done
        mock_task = MagicMock()
        mock_task.done.return_value = False
        _poc_tasks[app_id] = {
            "gen_task": mock_task,
            "stop_event": asyncio.Event(),
            "config": {"block_hash": "existing"},
            "stats": {},
        }
        
        # /generate with wait=false should still accept and queue
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "nonces": [0, 1],
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
            "wait": False,
        })
        
        assert response.status_code == 200
        assert response.json()["status"] == "queued"

    def test_generate_wait_false_returns_queued(self, client, mock_engine_client):
        """Test /generate with wait=false returns queued status."""
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "nonces": [0, 1, 2],
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
            "wait": False,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert data["queued_count"] == 3

    def test_generate_with_validation_detects_mismatch(self, client, mock_engine_client):
        """Test /generate with validation field performs comparison."""
        # 12-dim vectors: all zeros vs all ones (very different)
        mock_engine_client.poc_request.return_value = {
            "artifacts": [
                {"nonce": 0, "vector_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},
                {"nonce": 1, "vector_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},
            ],
        }
        
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "nonces": [0, 1],
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
            "wait": True,
            "validation": {
                "artifacts": [
                    {"nonce": 0, "vector_b64": "ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8"},  # ones - different
                    {"nonce": 1, "vector_b64": "ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8"},
                ],
            },
            "stat_test": {
                "dist_threshold": 0.02,
                "p_mismatch": 0.001,
                "fraud_threshold": 0.01,
            },
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["n_mismatch"] == 2
        assert data["fraud_detected"] is True

    def test_generate_validation_nonce_mismatch_error(self, client, mock_engine_client):
        """Test /generate returns 400 if validation nonces don't match."""
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "nonces": [0, 1],  # Two nonces
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
            "wait": True,
            "validation": {
                "artifacts": [
                    {"nonce": 0, "vector_b64": "AAA="},
                    {"nonce": 5, "vector_b64": "BBB="},  # Wrong nonce
                ],
            },
        })
        
        assert response.status_code == 400
        assert "must match" in response.json()["detail"]


class TestPoCStatus:
    def test_get_status_idle(self, client, mock_engine_client):
        """Test getting PoC status when idle (API-owned state)."""
        response = client.get("/api/v1/pow/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "IDLE"
        assert data["config"] is None
        assert data["stats"] is None

    def test_get_status_generating(self, client, app_with_poc, mock_engine_client):
        """Test getting PoC status when generating (API-owned state)."""
        import time
        app_id = id(app_with_poc)
        
        # Simulate active generation with config and stats
        mock_task = MagicMock()
        mock_task.done.return_value = False
        _poc_tasks[app_id] = {
            "gen_task": mock_task,
            "stop_event": asyncio.Event(),
            "config": {
                "block_hash": "abc123",
                "block_height": 100,
                "public_key": "pubkey123",
                "node_id": 0,
                "node_count": 1,
                "seq_len": 256,
                "k_dim": 12,
            },
            "stats": {
                "start_time": time.time(),
                "total_processed": 500,
            },
        }
        
        # Now get status
        response = client.get("/api/v1/pow/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "GENERATING"
        assert data["config"]["block_hash"] == "abc123"
        assert data["config"]["k_dim"] == 12
        assert data["stats"]["total_processed"] == 500


class TestPoCStop:
    def test_stop_round(self, client, mock_engine_client):
        """Test stopping a PoC round (API-only, no engine call)."""
        # Start generation first
        mock_engine_client.poc_request.return_value = {"artifacts": []}
        
        response1 = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
        })
        assert response1.status_code == 200
        
        # Stop it
        response2 = client.post("/api/v1/pow/stop")
        
        assert response2.status_code == 200
        data = response2.json()
        assert data["status"] == "OK"
        assert data["pow_status"]["status"] == "STOPPED"
        
        # Verify status is now IDLE
        response3 = client.get("/api/v1/pow/status")
        assert response3.json()["status"] == "IDLE"


class TestPoCDisabled:
    def test_routes_disabled_without_flag(self):
        """Test that routes return 503 when PoC is not enabled."""
        app = FastAPI()
        app.include_router(router)
        
        app.state.poc_enabled = False
        app.state.engine_client = AsyncMock()
        
        test_client = TestClient(app)
        
        response = test_client.get("/api/v1/pow/status")
        assert response.status_code == 503
        assert "not enabled" in response.json()["detail"]

    def test_routes_require_engine(self):
        """Test that routes return 503 when engine is not available."""
        app = FastAPI()
        app.include_router(router)
        
        app.state.poc_enabled = True
        # No engine_client
        
        test_client = TestClient(app)
        
        # /generate requires engine, /status doesn't (API-owned)
        response = test_client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "nonces": [0],
            "params": {"model": "test", "seq_len": 256, "k_dim": 12},
            "wait": True,
        })
        assert response.status_code == 503
        assert "not available" in response.json()["detail"]


class TestRemovedEndpoints:
    """Test that old endpoints are removed (404)."""
    
    def test_old_init_removed(self, client):
        """Test /init endpoint no longer exists."""
        response = client.post("/api/v1/pow/init", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "r_target": 0.5,
        })
        assert response.status_code == 404

    def test_old_init_validate_removed(self, client):
        """Test /init/validate endpoint no longer exists."""
        response = client.post("/api/v1/pow/init/validate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "r_target": 0.5,
        })
        assert response.status_code == 404

    def test_old_phase_generate_removed(self, client):
        """Test /phase/generate endpoint no longer exists."""
        response = client.post("/api/v1/pow/phase/generate")
        assert response.status_code == 404

    def test_old_phase_validate_removed(self, client):
        """Test /phase/validate endpoint no longer exists."""
        response = client.post("/api/v1/pow/phase/validate")
        assert response.status_code == 404

    def test_old_validate_removed(self, client):
        """Test /validate endpoint no longer exists."""
        response = client.post("/api/v1/pow/validate", json={
            "public_key": "pubkey123",
            "block_hash": "abc123",
            "block_height": 100,
            "nonces": [1, 2, 3],
            "dist": [0.1, 0.2, 0.3],
            "node_id": 0,
        })
        assert response.status_code == 404


class TestGenerationLoop:
    """Test generation loop behavior."""
    
    def test_nonce_generation_api_side(self):
        """Test that nonces are generated correctly in API."""
        from vllm.poc.routes import _get_next_nonces
        
        # Node 0 of 3: 0, 3, 6, 9
        nonces, counter = _get_next_nonces(nonce_counter=0, batch_size=4, node_count=3)
        assert nonces == [0, 3, 6, 9]
        assert counter == 12
        
        # Continue: 12, 15, 18, 21
        nonces2, counter2 = _get_next_nonces(nonce_counter=counter, batch_size=4, node_count=3)
        assert nonces2 == [12, 15, 18, 21]
        assert counter2 == 24
        
        # Node 1 of 3: 1, 4, 7, 10
        nonces3, counter3 = _get_next_nonces(nonce_counter=1, batch_size=4, node_count=3)
        assert nonces3 == [1, 4, 7, 10]
        
        # Single node: 0, 1, 2, 3
        nonces4, counter4 = _get_next_nonces(nonce_counter=0, batch_size=4, node_count=1)
        assert nonces4 == [0, 1, 2, 3]
        assert counter4 == 4


class TestGenerateQueue:
    """Test generate queue infrastructure."""
    
    def test_wait_false_returns_request_id(self, client, mock_engine_client):
        """Test wait=false returns a request_id that can be polled."""
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "nonces": [0, 1, 2],
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
            "wait": False,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert "request_id" in data
        assert data["queued_count"] == 3
    
    def test_poll_unknown_request_returns_404(self, client):
        """Test polling unknown request_id returns 404."""
        response = client.get("/api/v1/pow/generate/unknown-request-id")
        assert response.status_code == 404
    
    def test_poll_queued_request_returns_status(self, client, mock_engine_client):
        """Test polling a queued request returns its status."""
        # Enqueue a request
        response = client.post("/api/v1/pow/generate", json={
            "block_hash": "abc123",
            "block_height": 100,
            "public_key": "pubkey123",
            "node_id": 0,
            "node_count": 1,
            "nonces": [0],
            "params": {
                "model": "test-model",
                "seq_len": 256,
                "k_dim": 12,
            },
            "wait": False,
        })
        
        request_id = response.json()["request_id"]
        
        # Poll should return queued (or running if worker is fast)
        poll_response = client.get(f"/api/v1/pow/generate/{request_id}")
        assert poll_response.status_code == 200
        data = poll_response.json()
        assert data["status"] in ["queued", "running", "completed"]
        assert data["request_id"] == request_id
    
    def test_result_record_creation(self):
        """Test GenerateResult dataclass."""
        record = GenerateResult(status="queued")
        assert record.status == "queued"
        assert record.created_at > 0
        assert record.completed_at is None
        assert record.result is None
        assert record.error is None
    
    def test_cleanup_old_results(self):
        """Test that old results are cleaned up."""
        _generate_results.clear()
        
        # Add an old completed result
        old_time = time.time() - GENERATE_RESULT_TTL_SEC - 100
        _generate_results["old-id"] = GenerateResult(
            status="completed",
            created_at=old_time,
            completed_at=old_time + 1,
            result={"status": "completed"},
        )
        
        # Add a recent result
        _generate_results["new-id"] = GenerateResult(
            status="completed",
            created_at=time.time(),
            completed_at=time.time(),
            result={"status": "completed"},
        )
        
        # Cleanup
        _cleanup_old_results()
        
        # Old should be removed, new should remain
        assert "old-id" not in _generate_results
        assert "new-id" in _generate_results
        
        # Clean up test state
        _generate_results.clear()


class TestGenerateQueueIntegration:
    """Integration tests for generate queue with worker."""
    
    @pytest.mark.asyncio
    async def test_process_generate_job_computes_artifacts(self):
        """Test that _process_generate_job correctly processes a job."""
        from vllm.poc.routes import _process_generate_job, GenerateJob
        
        # Create a mock engine client
        mock_client = AsyncMock()
        mock_client.poc_request.return_value = {
            "artifacts": [
                {"nonce": 0, "vector_b64": "AAAA"},
                {"nonce": 1, "vector_b64": "BBBB"},
            ],
        }
        
        job = GenerateJob(
            request_id="test-job-1",
            engine_client=mock_client,
            app_id=12345,
            block_hash="abc123",
            block_height=100,
            public_key="pubkey",
            node_id=0,
            node_count=1,
            nonces=[0, 1],
            seq_len=256,
            k_dim=12,
            batch_size=10,
        )
        
        result = await _process_generate_job(job)
        
        assert result["status"] == "completed"
        assert len(result["artifacts"]) == 2
        assert result["encoding"]["dtype"] == "f16"
    
    @pytest.mark.asyncio
    async def test_process_generate_job_with_validation(self):
        """Test that _process_generate_job handles validation mode."""
        from vllm.poc.routes import _process_generate_job, GenerateJob
        
        # Mock returns zeros
        mock_client = AsyncMock()
        mock_client.poc_request.return_value = {
            "artifacts": [
                {"nonce": 0, "vector_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"},
            ],
        }
        
        # Validation expects ones (mismatch)
        job = GenerateJob(
            request_id="test-job-2",
            engine_client=mock_client,
            app_id=12345,
            block_hash="abc123",
            block_height=100,
            public_key="pubkey",
            node_id=0,
            node_count=1,
            nonces=[0],
            seq_len=256,
            k_dim=12,
            batch_size=10,
            validation_artifacts={0: "ADwAPAA8ADwAPAA8ADwAPAA8ADwAPAA8"},  # ones
        )
        
        result = await _process_generate_job(job)
        
        assert result["status"] == "completed"
        assert result["n_mismatch"] == 1
        assert result["fraud_detected"] is True
