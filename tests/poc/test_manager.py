import pytest
import time
import torch
from unittest.mock import Mock, patch, MagicMock

from vllm.poc.config import PoCConfig, PoCState
from vllm.poc.manager import PoCManager, PoCStats


class MockModelConfig:
    """Mock vLLM model config"""
    
    def get_hidden_size(self):
        return 128
    
    def get_vocab_size(self):
        return 1000


def create_mock_model_executor():
    """Create a mock model executor with driver_worker.
    
    Returns a mock executor that has:
    - driver_worker.device: torch.device
    - collective_rpc: Mock that can be configured
    """
    executor = MagicMock()
    executor.driver_worker = MagicMock()
    executor.driver_worker.device = torch.device("cpu")
    executor.collective_rpc = MagicMock(return_value=[None])
    return executor


def create_mock_vllm_config():
    """Create a mock VllmConfig for testing.
    
    The manager uses vllm_config for set_forward_context(), which needs:
    - compilation_config.static_forward_context: dict of attention layers
    - parallel_config.data_parallel_size: for DP metadata (we set to 1 to skip)
    """
    vllm_config = MagicMock()
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.parallel_config.data_parallel_size = 1
    return vllm_config


class TestPoCStats:
    def test_initial_state(self):
        stats = PoCStats()
        assert stats.total_checked == 0
        assert stats.total_valid == 0
        assert stats.elapsed == 0.0
        assert stats.rate == 0.0
    
    def test_elapsed_calculation(self):
        stats = PoCStats(total_checked=100, start_time=time.time() - 10)
        assert stats.elapsed >= 10
        assert stats.elapsed < 11
    
    def test_rate_calculation(self):
        stats = PoCStats(total_checked=100, start_time=time.time() - 10)
        assert stats.rate >= 9
        assert stats.rate <= 11


class TestPoCManagerInit:
    @pytest.fixture
    def mock_executor(self):
        return create_mock_model_executor()
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    def test_initial_state(self, manager):
        assert manager.state == PoCState.IDLE
        assert manager.config is None
        assert manager.valid_nonces == []
        assert manager.valid_distances == []


class TestPoCManagerStateTransitions:
    @pytest.fixture
    def mock_executor(self):
        return create_mock_model_executor()
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    @pytest.fixture
    def config(self):
        return PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
        )
    
    def test_init_round_sets_config(self, manager, config):
        manager.init_round(config)
        
        assert manager.state == PoCState.IDLE
        assert manager.config == config
    
    def test_start_generate_sets_state(self, manager, config):
        manager.init_round(config)
        
        manager.start_generate()
        assert manager.state == PoCState.GENERATING
    
    def test_start_validate_sets_state(self, manager, config):
        manager.init_round(config)
        
        manager.start_validate()
        assert manager.state == PoCState.VALIDATING
    
    def test_generating_to_validating_transition(self, manager, config):
        """Test GENERATING -> VALIDATING transition."""
        manager.init_round(config)
        
        manager.start_generate()
        assert manager.state == PoCState.GENERATING
        
        manager.start_validate()
        assert manager.state == PoCState.VALIDATING
    
    def test_init_round_resets_counters(self, manager, config):
        manager.valid_nonces = [1, 2, 3]
        manager.valid_distances = [0.1, 0.2, 0.3]
        manager.stats.total_checked = 100
        
        manager.init_round(config)
        
        assert manager.valid_nonces == []
        assert manager.valid_distances == []
        assert manager.stats.total_checked == 0
    
    def test_init_round_raises_if_already_generating(self, manager, config):
        manager.init_round(config)
        manager.start_generate()
        
        with pytest.raises(RuntimeError, match="Round already in progress"):
            manager.init_round(config)
    
    def test_start_generate_requires_init(self, manager):
        with pytest.raises(RuntimeError, match="Round not initialized"):
            manager.start_generate()
    
    def test_start_validate_requires_init(self, manager):
        with pytest.raises(RuntimeError, match="Round not initialized"):
            manager.start_validate()
    
    def test_stop_round_from_generating(self, manager, config):
        manager.init_round(config)
        manager.start_generate()
        
        manager.stop_round()
        assert manager.state == PoCState.STOPPED
    
    def test_stop_round_from_validating(self, manager, config):
        manager.init_round(config)
        manager.start_validate()
        
        manager.stop_round()
        assert manager.state == PoCState.STOPPED


class TestPoCManagerNonceGeneration:
    """Cross-check: Nonce iteration pattern with original NonceIterator"""
    
    @pytest.fixture
    def mock_executor(self):
        return create_mock_model_executor()
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    def test_single_node_nonces(self, manager):
        """Single node gets sequential nonces: 0, 1, 2, ..."""
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
            node_id=0,
            node_count=1,
            batch_size=4,
        )
        manager.init_round(config)
        manager.start_generate()
        
        nonces1 = manager.get_next_nonces()
        assert nonces1 == [0, 1, 2, 3]
        
        nonces2 = manager.get_next_nonces()
        assert nonces2 == [4, 5, 6, 7]
    
    def test_multi_node_nonces_node0(self, manager):
        """Node 0 of 3 gets: 0, 3, 6, 9, ..."""
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
            node_id=0,
            node_count=3,
            batch_size=4,
        )
        manager.init_round(config)
        manager.start_generate()
        
        nonces = manager.get_next_nonces()
        assert nonces == [0, 3, 6, 9]
    
    def test_multi_node_nonces_node1(self, manager):
        """Node 1 of 3 gets: 1, 4, 7, 10, ..."""
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
            node_id=1,
            node_count=3,
            batch_size=4,
        )
        manager.init_round(config)
        manager.start_generate()
        
        nonces = manager.get_next_nonces()
        assert nonces == [1, 4, 7, 10]
    
    def test_multi_node_nonces_node2(self, manager):
        """Node 2 of 3 gets: 2, 5, 8, 11, ..."""
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
            node_id=2,
            node_count=3,
            batch_size=4,
        )
        manager.init_round(config)
        manager.start_generate()
        
        nonces = manager.get_next_nonces()
        assert nonces == [2, 5, 8, 11]


class TestPoCManagerStatus:
    @pytest.fixture
    def mock_executor(self):
        return create_mock_model_executor()
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    def test_get_status_idle(self, manager):
        status = manager.get_status()
        assert status["state"] == "IDLE"
        assert status["valid_nonces"] == []
        assert status["valid_distances"] == []
        assert status["total_checked"] == 0
        assert status["total_valid"] == 0
    
    def test_get_status_with_data(self, manager):
        manager.valid_nonces = [1, 2, 3]
        manager.valid_distances = [0.1, 0.2, 0.3]
        manager.stats.total_checked = 100
        manager.stats.total_valid = 3
        
        status = manager.get_status()
        assert status["valid_nonces"] == [1, 2, 3]
        assert status["valid_distances"] == [0.1, 0.2, 0.3]
        assert status["total_checked"] == 100
        assert status["total_valid"] == 3


class TestPoCManagerBatch:
    """Test run_batch with mocked collective_rpc."""
    
    @pytest.fixture
    def mock_executor(self):
        executor = create_mock_model_executor()
        # Configure collective_rpc to return a result from "last PP rank"
        executor.collective_rpc.return_value = [
            None,  # First PP rank
            {  # Last PP rank
                "nonces": [0, 1, 2, 3],
                "distances": [0.1, 0.2, 0.3, 0.4],
            }
        ]
        return executor
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    def test_run_batch_calls_collective_rpc(self, manager, mock_executor):
        """run_batch should call collective_rpc with execute_poc_forward."""
        config = PoCConfig(
            block_hash="test_hash",
            block_height=100,
            public_key="test_node",
            r_target=0.5,
            batch_size=4,
            seq_len=32,
        )
        manager.init_round(config)
        manager.start_generate()
        
        batch = manager.run_batch()
        
        # Verify collective_rpc was called (once for forward)
        assert mock_executor.collective_rpc.call_count == 1
        # Call should be execute_poc_forward
        from vllm.poc.poc_model_runner import execute_poc_forward
        last_call = mock_executor.collective_rpc.call_args_list[-1]
        assert last_call[0][0] == execute_poc_forward
        
        # Verify batch data
        assert batch.nonces == [0, 1, 2, 3]
        assert batch.dist == [0.1, 0.2, 0.3, 0.4]
    
    def test_run_batch_updates_stats(self, manager, mock_executor):
        """run_batch should update stats."""
        config = PoCConfig(
            block_hash="test_hash",
            block_height=100,
            public_key="test_node",
            r_target=0.5,
            batch_size=4,
            seq_len=32,
        )
        manager.init_round(config)
        manager.start_generate()
        
        assert manager.stats.total_checked == 0
        
        manager.run_batch()
        
        assert manager.stats.total_checked == 4
    
    def test_run_batch_tracks_valid_nonces(self, manager, mock_executor):
        """run_batch should track valid nonces (d < r_target)."""
        # Set r_target so some are valid
        config = PoCConfig(
            block_hash="test_hash",
            block_height=100,
            public_key="test_node",
            r_target=0.25,  # First two will be valid (0.1, 0.2)
            batch_size=4,
            seq_len=32,
        )
        manager.init_round(config)
        manager.start_generate()
        
        manager.run_batch()
        
        assert manager.valid_nonces == [0, 1]
        assert manager.valid_distances == [0.1, 0.2]
        assert manager.stats.total_valid == 2
    
    def test_run_batch_returns_empty_when_not_generating(self, manager):
        """run_batch returns empty batch if not in GENERATING state."""
        config = PoCConfig(
            block_hash="test_hash",
            block_height=100,
            public_key="test_node",
            r_target=0.5,
        )
        manager.init_round(config)
        # Don't call start_generate()
        
        batch = manager.run_batch()
        
        assert len(batch) == 0
    
    def test_run_batch_with_state_returns_full_result(self, manager, mock_executor):
        """run_batch_with_state returns batch + state for optimized loop."""
        config = PoCConfig(
            block_hash="test_hash",
            block_height=100,
            public_key="test_node",
            r_target=0.25,  # First two will be valid
            batch_size=4,
            seq_len=32,
            node_id=0,
        )
        manager.init_round(config)
        manager.start_generate()
        
        result = manager.run_batch_with_state()
        
        assert result["should_continue"] is True
        assert result["state"] == "GENERATING"
        assert result["public_key"] == "test_node"
        assert result["block_hash"] == "test_hash"
        assert result["block_height"] == 100
        assert result["node_id"] == 0
        assert result["nonces"] == [0, 1, 2, 3]
        assert result["distances"] == [0.1, 0.2, 0.3, 0.4]
        assert result["valid_nonces"] == [0, 1]  # d < 0.25
        assert result["valid_distances"] == [0.1, 0.2]
    
    def test_run_batch_with_state_returns_false_when_not_generating(self, manager):
        """run_batch_with_state returns should_continue=False when not generating."""
        config = PoCConfig(
            block_hash="test_hash",
            block_height=100,
            public_key="test_node",
            r_target=0.5,
        )
        manager.init_round(config)
        # Don't call start_generate()
        
        result = manager.run_batch_with_state()
        
        assert result["should_continue"] is False
        assert result["state"] == "IDLE"


class TestPoCManagerValidate:
    """Test validate with mocked collective_rpc."""
    
    @pytest.fixture
    def mock_executor(self):
        executor = create_mock_model_executor()
        return executor
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    def test_validate_calls_collective_rpc(self, manager, mock_executor):
        """validate should call collective_rpc with execute_poc_forward."""
        mock_executor.collective_rpc.return_value = [
            None,
            {
                "nonces": [0, 1, 2],
                "distances": [0.1, 0.6, 0.2],
            }
        ]
        
        config = PoCConfig(
            block_hash="test_hash",
            block_height=100,
            public_key="test_node",
            r_target=0.5,
        )
        manager.init_round(config)
        manager.start_validate()
        
        result = manager.validate([0, 1, 2], "test_node")
        
        # Verify collective_rpc was called (once for forward)
        assert mock_executor.collective_rpc.call_count == 1
        # Call should be execute_poc_forward
        from vllm.poc.poc_model_runner import execute_poc_forward
        last_call = mock_executor.collective_rpc.call_args_list[-1]
        assert last_call[0][0] == execute_poc_forward
        
        # Verify results (validate computes valid flags from distances)
        assert result["computed_distances"] == [0.1, 0.6, 0.2]
        assert result["valid"] == [True, False, True]
    
    def test_validate_requires_config(self, manager):
        """validate raises error if no round configured."""
        with pytest.raises(RuntimeError, match="No round configured"):
            manager.validate([0, 1, 2], "test_node")


# GPU Tests - require CUDA and load actual model
# These tests are skipped because running the model outside the worker
# context causes parallel group initialization errors. Proper GPU tests
# should be done in Phase 5 E2E tests where the full vLLM server runs.
@pytest.mark.gpu
@pytest.mark.skip(reason="GPU tests require full worker context - deferred to Phase 5 E2E tests")
class TestPoCManagerGPU:
    """GPU tests that load a real model to test run_batch and validate.
    
    WARNING: These tests are currently skipped because:
    - The model needs to run inside the vLLM worker process context
    - Extracting the model and running it directly causes parallel group errors
    - Proper testing should be done in Phase 5 with full E2E integration tests
    """
    pass
