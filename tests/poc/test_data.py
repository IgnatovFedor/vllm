from vllm.poc import PoCConfig, PoCState, ProofBatch, ValidatedBatch


class TestPoCConfig:
    def test_defaults(self):
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
        )
        assert config.batch_size == 32
        assert config.seq_len == 256
        assert config.fraud_threshold == 0.01
        assert config.node_id == 0
        assert config.node_count == 1
        assert config.callback_url is None

    def test_all_fields(self):
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
            fraud_threshold=0.05,
            node_id=2,
            node_count=4,
            batch_size=64,
            seq_len=128,
            callback_url="http://localhost:8080/callback",
        )
        assert config.node_id == 2
        assert config.node_count == 4
        assert config.callback_url == "http://localhost:8080/callback"


class TestPoCState:
    def test_states_exist(self):
        assert PoCState.IDLE.value == "IDLE"
        assert PoCState.GENERATING.value == "GENERATING"
        assert PoCState.VALIDATING.value == "VALIDATING"
        assert PoCState.STOPPED.value == "STOPPED"


class TestProofBatch:
    def test_empty(self):
        batch = ProofBatch.empty()
        assert len(batch) == 0
        assert batch.nonces == []
        assert batch.dist == []

    def test_len(self):
        batch = ProofBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2, 3],
            dist=[0.1, 0.2, 0.3],
            node_id=0,
        )
        assert len(batch) == 3

    def test_sub_batch_filters_correctly(self):
        batch = ProofBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2, 3, 4],
            dist=[0.3, 0.6, 0.4, 0.8],
            node_id=0,
        )
        filtered = batch.sub_batch(r_target=0.5)
        assert filtered.nonces == [1, 3]
        assert filtered.dist == [0.3, 0.4]
        assert filtered.public_key == "node1"
        assert filtered.block_hash == "hash1"

    def test_sub_batch_empty_result(self):
        batch = ProofBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2],
            dist=[0.6, 0.8],
            node_id=0,
        )
        filtered = batch.sub_batch(r_target=0.5)
        assert len(filtered) == 0

    def test_sub_batch_all_pass(self):
        batch = ProofBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2, 3],
            dist=[0.1, 0.2, 0.3],
            node_id=0,
        )
        filtered = batch.sub_batch(r_target=0.5)
        assert len(filtered) == 3


class TestValidatedBatch:
    def test_fraud_detection_all_valid(self):
        batch = ValidatedBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2, 3],
            received_dist=[0.3, 0.4, 0.45],
            dist=[0.3, 0.4, 0.45],
            r_target=0.5,
            fraud_threshold=0.01,
            node_id=0,
        )
        assert batch.n_invalid == 0
        assert batch.fraud_detected == False

    def test_fraud_detection_received_invalid(self):
        batch = ValidatedBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2, 3],
            received_dist=[0.3, 0.6, 0.45],
            dist=[0.3, 0.4, 0.45],
            r_target=0.5,
            fraud_threshold=0.01,
            node_id=0,
        )
        assert batch.n_invalid == 1

    def test_fraud_detection_computed_invalid(self):
        batch = ValidatedBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[1, 2, 3],
            received_dist=[0.3, 0.4, 0.45],
            dist=[0.3, 0.4, 0.55],
            r_target=0.5,
            fraud_threshold=0.01,
            node_id=0,
        )
        assert batch.n_invalid == 1

    def test_empty_batch_no_fraud(self):
        batch = ValidatedBatch(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            nonces=[],
            received_dist=[],
            dist=[],
            r_target=0.5,
            fraud_threshold=0.01,
            node_id=0,
        )
        assert batch.n_invalid == 0
        assert batch.fraud_detected == False

