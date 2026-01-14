"""Unit tests for layer_hooks context variable behavior."""
import pytest
import torch
from unittest.mock import Mock, MagicMock

from vllm.poc.layer_hooks import (
    poc_forward_context,
    is_poc_forward_active,
    LayerHouseholderHook,
)


class TestPoCForwardContext:
    """Tests for the poc_forward_context context manager."""
    
    def test_context_var_default_false(self):
        """Context should be False by default."""
        assert is_poc_forward_active() is False
    
    def test_context_manager_sets_true(self):
        """Context should be True inside context manager."""
        assert is_poc_forward_active() is False
        
        with poc_forward_context():
            assert is_poc_forward_active() is True
        
        assert is_poc_forward_active() is False
    
    def test_context_manager_exception_safety(self):
        """Context should reset even on exception."""
        assert is_poc_forward_active() is False
        
        try:
            with poc_forward_context():
                assert is_poc_forward_active() is True
                raise ValueError("test exception")
        except ValueError:
            pass
        
        assert is_poc_forward_active() is False
    
    def test_nested_context_managers(self):
        """Nested context managers should work correctly."""
        assert is_poc_forward_active() is False
        
        with poc_forward_context():
            assert is_poc_forward_active() is True
            with poc_forward_context():
                assert is_poc_forward_active() is True
            # Inner context resets, but outer still active
            assert is_poc_forward_active() is True
        
        assert is_poc_forward_active() is False


class TestLayerHouseholderHookContextAware:
    """Tests for context-aware hook behavior."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model with layers."""
        model = MagicMock()
        layer1 = MagicMock()
        layer2 = MagicMock()
        model.model.layers = [layer1, layer2]
        return model
    
    @pytest.fixture
    def hook_instance(self, mock_model):
        """Create a LayerHouseholderHook instance with setup called."""
        device = torch.device("cpu")
        hook = LayerHouseholderHook(
            model=mock_model,
            block_hash="test_block_hash",
            device=device,
            hidden_size=64,
        )
        # Manually call _setup since __init__ doesn't call it (lazy init)
        hook._setup(mock_model, "test_block_hash", device, 64)
        return hook
    
    def test_hook_passes_through_when_context_inactive(self, hook_instance):
        """Hook should pass through output unchanged when context is False."""
        assert is_poc_forward_active() is False
        
        # Get the hook function
        hook_fn = hook_instance._create_hook(0)
        
        # Create mock output
        mock_output = torch.randn(2, 10, 64)
        
        # Call hook outside of PoC context
        result = hook_fn(None, None, mock_output)
        
        # Should return original output unchanged
        assert result is mock_output
    
    def test_hook_transforms_when_context_active(self, hook_instance):
        """Hook should transform output when context is True."""
        # Get the hook function
        hook_fn = hook_instance._create_hook(0)
        
        # Create mock output
        mock_output = torch.randn(2, 10, 64)
        original_output = mock_output.clone()
        
        # Call hook inside PoC context
        with poc_forward_context():
            result = hook_fn(None, None, mock_output)
        
        # Should return transformed output (different from original)
        assert result is not mock_output
        assert not torch.allclose(result, original_output)
    
    def test_hook_handles_tuple_output_with_context(self, hook_instance):
        """Hook should handle tuple output correctly with context."""
        hook_fn = hook_instance._create_hook(0)
        
        # Create tuple output (hidden_states, residual)
        hidden = torch.randn(2, 10, 64)
        residual = torch.randn(2, 10, 64)
        mock_output = (hidden, residual)
        
        # Outside context - should pass through
        result_no_ctx = hook_fn(None, None, mock_output)
        assert result_no_ctx is mock_output
        
        # Inside context - should transform
        with poc_forward_context():
            result_with_ctx = hook_fn(None, None, (hidden.clone(), residual.clone()))
        
        assert isinstance(result_with_ctx, tuple)
        assert len(result_with_ctx) == 2
        # Both tensors should be transformed
        assert not torch.allclose(result_with_ctx[0], hidden)
        assert not torch.allclose(result_with_ctx[1], residual)
    
    def test_hook_deterministic_with_same_block_hash(self, mock_model):
        """Same block_hash should produce same transforms."""
        device = torch.device("cpu")
        
        hook1 = LayerHouseholderHook(mock_model, "same_hash", device, 64)
        hook2 = LayerHouseholderHook(mock_model, "same_hash", device, 64)
        
        # Reflection vectors should be identical
        for v1, v2 in zip(hook1.reflection_vectors, hook2.reflection_vectors):
            assert torch.allclose(v1, v2)
        
        hook1.detach()
        hook2.detach()
    
    def test_hook_different_with_different_block_hash(self, mock_model):
        """Different block_hash should produce different transforms."""
        device = torch.device("cpu")
        
        hook1 = LayerHouseholderHook(mock_model, "hash_A", device, 64)
        hook2 = LayerHouseholderHook(mock_model, "hash_B", device, 64)
        
        # Reflection vectors should be different
        for v1, v2 in zip(hook1.reflection_vectors, hook2.reflection_vectors):
            assert not torch.allclose(v1, v2)
        
        hook1.detach()
        hook2.detach()

