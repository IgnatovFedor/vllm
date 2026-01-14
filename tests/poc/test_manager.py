"""Tests for PoCManager (stateless artifact generation)."""
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock

from vllm.poc.manager import PoCManager
from vllm.poc.data import decode_vector


class MockModelConfig:
    """Mock vLLM model config"""
    
    def get_hidden_size(self):
        return 128
    
    def get_vocab_size(self):
        return 1000


def create_mock_model_executor():
    """Create a mock model executor with driver_worker."""
    executor = MagicMock()
    executor.driver_worker = MagicMock()
    executor.driver_worker.device = torch.device("cpu")
    executor.collective_rpc = MagicMock(return_value=[None])
    return executor


def create_mock_vllm_config():
    """Create a mock VllmConfig for testing."""
    vllm_config = MagicMock()
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.parallel_config.data_parallel_size = 1
    return vllm_config


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
        """Test manager initializes correctly (stateless)."""
        assert manager.model_executor is not None
        assert manager.model_config is not None
        assert manager.vllm_config is not None


class TestPoCManagerGenerateArtifacts:
    """Test generate_artifacts with mocked collective_rpc."""
    
    @pytest.fixture
    def mock_executor(self):
        executor = create_mock_model_executor()
        # Configure collective_rpc to return artifacts from "last PP rank"
        vectors = np.random.randn(4, 12).astype(np.float16)
        executor.collective_rpc.return_value = [
            None,  # Other ranks
            {"nonces": [0, 1, 2, 3], "vectors": vectors},  # Last PP rank
        ]
        return executor
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    def test_generate_artifacts_returns_artifacts(self, manager, mock_executor):
        """Test generate_artifacts returns artifacts list."""
        artifacts = manager.generate_artifacts(
            nonces=[0, 1, 2, 3],
            block_hash="hash1",
            public_key="node1",
            seq_len=256,
            k_dim=12,
        )
        
        assert len(artifacts) == 4
        assert artifacts[0].nonce == 0
        assert artifacts[3].nonce == 3
        
        # Verify collective_rpc was called
        mock_executor.collective_rpc.assert_called_once()
    
    def test_generate_artifacts_encodes_vectors(self, manager, mock_executor):
        """Test that vectors are properly encoded to base64."""
        artifacts = manager.generate_artifacts(
            nonces=[0, 1],
            block_hash="hash1",
            public_key="node1",
            seq_len=256,
            k_dim=12,
        )
        
        # Should be able to decode vectors back
        for artifact in artifacts:
            vector = decode_vector(artifact.vector_b64)
            assert len(vector) == 12  # k_dim
            assert vector.dtype == np.float32  # decode returns float32
    
    def test_generate_artifacts_empty_when_no_result(self, manager, mock_executor):
        """Test generate_artifacts returns empty list when forward returns None."""
        mock_executor.collective_rpc.return_value = [None, None]
        
        artifacts = manager.generate_artifacts(
            nonces=[0, 1],
            block_hash="hash1",
            public_key="node1",
            seq_len=256,
            k_dim=12,
        )
        
        assert artifacts == []
    
    def test_generate_artifacts_passes_correct_args(self, manager, mock_executor):
        """Test that generate_artifacts passes correct args to collective_rpc."""
        manager.generate_artifacts(
            nonces=[10, 20, 30],
            block_hash="test_hash",
            public_key="test_pubkey",
            seq_len=512,
            k_dim=16,
        )
        
        # Verify the args passed to collective_rpc
        call_args = mock_executor.collective_rpc.call_args
        assert call_args is not None
        
        # First positional arg is the function
        # args= tuple contains the actual arguments
        args_tuple = call_args.kwargs.get('args') or call_args[1].get('args')
        if args_tuple is None:
            args_tuple = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get('args', ())
        
        # The args should include block_hash, public_key, nonces, seq_len, hidden_size, k_dim
        # Order: (block_hash, public_key, nonces, seq_len, hidden_size, k_dim)
        assert "test_hash" in str(args_tuple)
        assert "test_pubkey" in str(args_tuple)


class TestPoCManagerStateless:
    """Test that PoCManager is stateless."""
    
    @pytest.fixture
    def mock_executor(self):
        executor = create_mock_model_executor()
        # Return dynamic nonces based on what was passed
        def mock_collective_rpc(func, args=None, **kwargs):
            if args:
                # args = (block_hash, public_key, nonces, seq_len, hidden_size, k_dim)
                nonces = args[2] if len(args) > 2 else [0, 1]
                vectors = np.random.randn(len(nonces), 12).astype(np.float16)
                return [{"nonces": nonces, "vectors": vectors}]
            return [None]
        executor.collective_rpc.side_effect = mock_collective_rpc
        return executor
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_executor, mock_vllm_config):
        return PoCManager(mock_executor, MockModelConfig(), mock_vllm_config)
    
    def test_no_state_between_calls(self, manager):
        """Test that manager doesn't maintain state between generate_artifacts calls."""
        # First call
        artifacts1 = manager.generate_artifacts(
            nonces=[0, 1],
            block_hash="hash1",
            public_key="node1",
            seq_len=256,
            k_dim=12,
        )
        
        # Second call with different params
        artifacts2 = manager.generate_artifacts(
            nonces=[100, 101],
            block_hash="hash2",
            public_key="node2",
            seq_len=512,
            k_dim=16,
        )
        
        # Both should succeed independently
        assert len(artifacts1) == 2
        assert len(artifacts2) == 2
        assert artifacts1[0].nonce == 0
        assert artifacts2[0].nonce == 100
