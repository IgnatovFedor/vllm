import time
import torch
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

from .config import PoCConfig, PoCState
from .data import ProofBatch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.executor.executor_base import ExecutorBase


@dataclass
class PoCStats:
    total_checked: int = 0
    total_valid: int = 0
    start_time: float = 0.0
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time > 0 else 0.0
    
    @property
    def rate(self) -> float:
        return self.total_checked / self.elapsed if self.elapsed > 0 else 0.0


class PoCManager:
    def __init__(
        self,
        model_executor: "ExecutorBase",
        model_config,
        vllm_config: "VllmConfig",
    ):
        self.model_executor = model_executor
        self.model_config = model_config
        self.vllm_config = vllm_config
        self.device = self._get_device()
        
        self.state = PoCState.IDLE
        self.config: Optional[PoCConfig] = None
        self.stats = PoCStats()
        
        self.valid_nonces: List[int] = []
        self.valid_distances: List[float] = []
        self._nonce_counter = 0
    
    def _get_device(self) -> torch.device:
        if hasattr(self.model_executor, 'driver_worker'):
            return self.model_executor.driver_worker.device
        return torch.device("cuda:0")
    
    def init_round(self, config: PoCConfig) -> None:
        if self.state == PoCState.GENERATING:
            raise RuntimeError("Round already in progress")
        
        self.config = config
        self.stats = PoCStats(start_time=time.time())
        self.valid_nonces = []
        self.valid_distances = []
        self._nonce_counter = config.node_id
        self.state = PoCState.IDLE
    
    def start_generate(self) -> None:
        if self.config is None:
            raise RuntimeError("Round not initialized")
        self.state = PoCState.GENERATING
    
    def start_validate(self) -> None:
        if self.config is None:
            raise RuntimeError("Round not initialized")
        self.state = PoCState.VALIDATING
    
    def stop_round(self) -> None:
        self.state = PoCState.STOPPED
    
    def get_next_nonces(self) -> List[int]:
        nonces = []
        for _ in range(self.config.batch_size):
            nonces.append(self._nonce_counter)
            self._nonce_counter += self.config.node_count
        return nonces
    
    def _run_forward(
        self,
        block_hash: str,
        public_key: str,
        nonces: List[int],
        seq_len: int,
        return_vectors: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Run forward pass via collective_rpc."""
        from .poc_model_runner import execute_poc_forward

        results = self.model_executor.collective_rpc(
            execute_poc_forward,
            args=(
                block_hash,
                public_key,
                nonces,
                seq_len,
                self.model_config.get_hidden_size(),
                self.config.r_target if self.config else 1.5,
                self.vllm_config,
                return_vectors,
            ),
        )
        
        # Only the last PP rank returns a result
        return next((r for r in results if r is not None), None)
    
    def run_batch(self) -> ProofBatch:
        if self.state != PoCState.GENERATING:
            return ProofBatch.empty()
        
        nonces = self.get_next_nonces()
        
        result = self._run_forward(
            self.config.block_hash,
            self.config.public_key,
            nonces,
            self.config.seq_len,
        )
        
        if result is None:
            return ProofBatch.empty()
        
        batch = ProofBatch(
            public_key=self.config.public_key,
            block_hash=self.config.block_hash,
            block_height=self.config.block_height,
            nonces=result["nonces"],
            dist=result["distances"],
            node_id=self.config.node_id,
        )
        
        self.stats.total_checked += len(nonces)
        
        valid_batch = batch.sub_batch(self.config.r_target)
        self.stats.total_valid += len(valid_batch)
        self.valid_nonces.extend(valid_batch.nonces)
        self.valid_distances.extend(valid_batch.dist)
        
        return batch
    
    def run_batch_with_state(self) -> dict:
        if self.state != PoCState.GENERATING:
            return {
                "should_continue": False,
                "state": self.state.value,
            }
        
        batch = self.run_batch()
        valid_batch = batch.sub_batch(self.config.r_target)
        
        return {
            "should_continue": self.state == PoCState.GENERATING,
            "state": self.state.value,
            "public_key": self.config.public_key,
            "block_hash": self.config.block_hash,
            "block_height": self.config.block_height,
            "node_id": self.config.node_id,
            "nonces": batch.nonces,
            "distances": batch.dist,
            "valid_nonces": valid_batch.nonces,
            "valid_distances": valid_batch.dist,
        }
    
    def validate(self, nonces: List[int], public_key: str, 
                 received_dist: Optional[List[float]] = None) -> Dict[str, Any]:
        if self.config is None:
            raise RuntimeError("No round configured")
        
        result = self._run_forward(
            self.config.block_hash,
            public_key,
            nonces,
            self.config.seq_len,
        )
        
        if result is None:
            raise RuntimeError("No result from validation")
        
        computed_dist = result["distances"]
        valid_flags = [d < self.config.r_target for d in computed_dist]
        
        self.stats.total_checked += len(nonces)
        self.stats.total_valid += sum(valid_flags)
        
        fraud_detected = False
        if received_dist is not None and len(received_dist) == len(computed_dist):
            for recv, comp in zip(received_dist, computed_dist):
                if recv < self.config.r_target and comp >= self.config.r_target:
                    fraud_detected = True
                    break
        
        return {
            "nonces": nonces,
            "computed_distances": computed_dist,
            "received_distances": received_dist or [],
            "valid": valid_flags,
            "fraud_detected": fraud_detected,
            "public_key": public_key,
            "block_hash": self.config.block_hash,
            "block_height": self.config.block_height,
        }
    
    def generate_for_nonces(
        self,
        nonces: List[int],
        block_hash: str,
        public_key: str,
        r_target: float,
        seq_len: int,
        return_vectors: bool = False,
    ) -> Dict[str, Any]:
        """Generate distances for specific nonces."""
        # Temporarily set r_target for this call
        old_r_target = self.config.r_target if self.config else None
        if self.config:
            self.config.r_target = r_target
        
        result = self._run_forward(
            block_hash,
            public_key,
            nonces,
            seq_len,
            return_vectors,
        )
        
        # Restore r_target
        if self.config and old_r_target is not None:
            self.config.r_target = old_r_target
        
        if result is None:
            return {"nonces": nonces, "distances": []}
        
        response = {
            "nonces": result["nonces"],
            "distances": result["distances"],
        }
        if return_vectors and "vectors" in result:
            response["vectors"] = result["vectors"]
        return response
    
    def teardown_generate_hooks(self) -> None:
        """No-op, kept for API compatibility."""
        pass
    
    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "valid_nonces": self.valid_nonces,
            "valid_distances": self.valid_distances,
            "total_checked": self.stats.total_checked,
            "total_valid": self.stats.total_valid,
            "elapsed_seconds": self.stats.elapsed,
            "rate_per_second": self.stats.rate,
            "r_target": self.config.r_target if self.config else None,
        }
