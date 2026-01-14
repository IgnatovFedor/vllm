"""Tests for PoC data types and helpers (artifact-based protocol)."""
import numpy as np
import pytest

from vllm.poc import (
    PoCConfig, PoCState, PoCParams,
    Artifact, Encoding, ArtifactBatch, ValidationResult,
    encode_vector, decode_vector,
    is_mismatch, fraud_test, compare_artifacts,
)


class TestPoCConfig:
    def test_defaults(self):
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
        )
        assert config.batch_size == 32
        assert config.seq_len == 256
        assert config.k_dim == 12
        assert config.node_id == 0
        assert config.node_count == 1
        assert config.callback_url is None

    def test_all_fields(self):
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            node_id=2,
            node_count=4,
            batch_size=64,
            seq_len=128,
            k_dim=8,
            callback_url="http://localhost:8080/callback",
        )
        assert config.node_id == 2
        assert config.node_count == 4
        assert config.k_dim == 8
        assert config.callback_url == "http://localhost:8080/callback"


class TestPoCState:
    def test_states_exist(self):
        assert PoCState.IDLE.value == "IDLE"
        assert PoCState.GENERATING.value == "GENERATING"
        assert PoCState.STOPPED.value == "STOPPED"


class TestPoCParams:
    def test_creation(self):
        params = PoCParams(model="Qwen/Qwen3-0.6B", seq_len=256, k_dim=12)
        assert params.model == "Qwen/Qwen3-0.6B"
        assert params.seq_len == 256
        assert params.k_dim == 12
    
    def test_default_k_dim(self):
        params = PoCParams(model="test-model", seq_len=128)
        assert params.k_dim == 12


class TestArtifact:
    def test_creation(self):
        artifact = Artifact(nonce=42, vector_b64="AAAAAAAAAAAAAAAA")
        assert artifact.nonce == 42
        assert artifact.vector_b64 == "AAAAAAAAAAAAAAAA"


class TestEncoding:
    def test_defaults(self):
        encoding = Encoding()
        assert encoding.dtype == "f16"
        assert encoding.k_dim == 12
        assert encoding.endian == "le"
    
    def test_custom(self):
        encoding = Encoding(dtype="f32", k_dim=8, endian="be")
        assert encoding.dtype == "f32"
        assert encoding.k_dim == 8
        assert encoding.endian == "be"


class TestVectorEncoding:
    def test_encode_decode_roundtrip(self):
        """Encode FP32 vector to base64, decode back to FP32."""
        original = np.array([0.1, 0.2, 0.3, -0.5, 1.0], dtype=np.float32)
        encoded = encode_vector(original)
        decoded = decode_vector(encoded)
        
        # Should be very close (FP16 may lose some precision)
        np.testing.assert_allclose(decoded, original, rtol=1e-3)
    
    def test_encode_produces_base64(self):
        """Encoded string should be valid base64."""
        import base64
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        encoded = encode_vector(vec)
        
        # Should be decodable as base64
        decoded_bytes = base64.b64decode(encoded)
        assert len(decoded_bytes) == 3 * 2  # 3 floats * 2 bytes each (FP16)
    
    def test_decode_correct_length(self):
        """Decoded vector should have correct length."""
        # Encode a 12-dim vector
        vec = np.random.randn(12).astype(np.float32)
        encoded = encode_vector(vec)
        decoded = decode_vector(encoded)
        
        assert decoded.shape == (12,)
    
    def test_little_endian_format(self):
        """Verify little-endian byte order."""
        import struct
        
        # Known value
        vec = np.array([1.0], dtype=np.float32)
        encoded = encode_vector(vec)
        decoded_bytes = np.frombuffer(
            __import__('base64').b64decode(encoded), 
            dtype='<f2'
        )
        
        # 1.0 in FP16 little-endian
        assert decoded_bytes[0] == np.float16(1.0)


class TestIsMismatch:
    def test_identical_vectors_no_mismatch(self):
        """Identical vectors should not be a mismatch."""
        vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        b64 = encode_vector(vec)
        
        assert is_mismatch(vec, b64, dist_threshold=0.01) is False
    
    def test_different_vectors_is_mismatch(self):
        """Clearly different vectors should be a mismatch."""
        computed = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        received = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b64 = encode_vector(received)
        
        assert is_mismatch(computed, b64, dist_threshold=0.01) is True
    
    def test_threshold_boundary(self):
        """Test behavior at threshold boundary."""
        vec1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.015, 0.0, 0.0], dtype=np.float32)  # L2 dist = 0.015
        b64 = encode_vector(vec2)
        
        # Below threshold
        assert is_mismatch(vec1, b64, dist_threshold=0.02) is False
        # Above threshold
        assert is_mismatch(vec1, b64, dist_threshold=0.01) is True


class TestFraudTest:
    def test_no_mismatch_no_fraud(self):
        """Zero mismatches should not detect fraud."""
        p_value, fraud_detected = fraud_test(
            n_mismatch=0, n_total=100,
            p_mismatch=0.001, fraud_threshold=0.01
        )
        
        assert fraud_detected is False
        assert p_value > 0.01
    
    def test_high_mismatch_detects_fraud(self):
        """Many mismatches should detect fraud."""
        p_value, fraud_detected = fraud_test(
            n_mismatch=50, n_total=100,  # 50% mismatch rate
            p_mismatch=0.001, fraud_threshold=0.01
        )
        
        assert fraud_detected is True
        assert p_value < 0.01
    
    def test_empty_no_fraud(self):
        """Empty batch should not detect fraud."""
        p_value, fraud_detected = fraud_test(
            n_mismatch=0, n_total=0,
            p_mismatch=0.001, fraud_threshold=0.01
        )
        
        assert fraud_detected is False
        assert p_value == 1.0
    
    def test_single_mismatch_small_sample(self):
        """Single mismatch in small sample may or may not be fraud."""
        p_value, fraud_detected = fraud_test(
            n_mismatch=1, n_total=10,
            p_mismatch=0.001, fraud_threshold=0.01
        )
        
        # With p_mismatch=0.001 and 1/10 mismatches, p_value should be low
        assert p_value < 0.1


class TestCompareArtifacts:
    def test_identical_artifacts_no_mismatch(self):
        """Comparing identical artifacts should have no mismatches."""
        vec1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vec2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        
        computed = [vec1, vec2]
        artifacts = [
            Artifact(nonce=0, vector_b64=encode_vector(vec1)),
            Artifact(nonce=1, vector_b64=encode_vector(vec2)),
        ]
        
        n_mismatch, mismatch_nonces = compare_artifacts(
            computed, artifacts, dist_threshold=0.01
        )
        
        assert n_mismatch == 0
        assert mismatch_nonces == []
    
    def test_different_artifacts_has_mismatch(self):
        """Comparing different artifacts should detect mismatches."""
        vec1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vec2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        
        computed = [vec1, vec2]
        
        # Second artifact has wrong vector
        wrong_vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        artifacts = [
            Artifact(nonce=0, vector_b64=encode_vector(vec1)),
            Artifact(nonce=1, vector_b64=encode_vector(wrong_vec)),
        ]
        
        n_mismatch, mismatch_nonces = compare_artifacts(
            computed, artifacts, dist_threshold=0.01
        )
        
        assert n_mismatch == 1
        assert mismatch_nonces == [1]
    
    def test_all_mismatched(self):
        """All different should report all as mismatches."""
        computed = [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        ]
        artifacts = [
            Artifact(nonce=5, vector_b64=encode_vector(np.array([1.0, 0.0], dtype=np.float32))),
            Artifact(nonce=10, vector_b64=encode_vector(np.array([0.0, 1.0], dtype=np.float32))),
        ]
        
        n_mismatch, mismatch_nonces = compare_artifacts(
            computed, artifacts, dist_threshold=0.01
        )
        
        assert n_mismatch == 2
        assert set(mismatch_nonces) == {5, 10}


class TestArtifactBatch:
    def test_creation(self):
        batch = ArtifactBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            node_id=0,
            artifacts=[Artifact(nonce=0, vector_b64="abc")],
            encoding=Encoding(),
        )
        
        assert batch.public_key == "node1"
        assert len(batch.artifacts) == 1


class TestValidationResult:
    def test_creation(self):
        result = ValidationResult(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            node_id=0,
            nonces=[1, 2, 3],
            n_total=3,
            n_mismatch=1,
            mismatch_nonces=[2],
        )
        
        assert result.n_total == 3
        assert result.n_mismatch == 1
        assert result.mismatch_nonces == [2]
