from dataclasses import dataclass, field
from typing import List

from scipy.stats import binomtest

PROBABILITY_MISMATCH = 5e-4


@dataclass
class ProofBatch:
    public_key: str
    block_hash: str
    block_height: int
    nonces: List[int]
    dist: List[float]
    node_id: int

    def sub_batch(self, r_target: float) -> 'ProofBatch':
        mask = [d < r_target for d in self.dist]
        return ProofBatch(
            public_key=self.public_key,
            block_hash=self.block_hash,
            block_height=self.block_height,
            nonces=[n for n, m in zip(self.nonces, mask) if m],
            dist=[d for d, m in zip(self.dist, mask) if m],
            node_id=self.node_id,
        )

    def __len__(self) -> int:
        return len(self.nonces)

    @staticmethod
    def empty() -> 'ProofBatch':
        return ProofBatch(
            public_key="",
            block_hash="",
            block_height=-1,
            nonces=[],
            dist=[],
            node_id=-1,
        )


@dataclass
class ValidatedBatch:
    public_key: str
    block_hash: str
    block_height: int
    nonces: List[int]
    received_dist: List[float]
    dist: List[float]
    r_target: float
    fraud_threshold: float
    node_id: int
    n_invalid: int = field(default=-1)
    probability_honest: float = field(default=-1.0)
    fraud_detected: bool = field(default=False)

    def __post_init__(self):
        if self.n_invalid >= 0:
            return
        self.n_invalid = sum(
            1 for rd, cd in zip(self.received_dist, self.dist)
            if rd >= self.r_target or cd > self.r_target
        )
        if len(self.nonces) > 0:
            self.probability_honest = float(
                binomtest(
                    k=self.n_invalid,
                    n=len(self.nonces),
                    p=PROBABILITY_MISMATCH,
                    alternative='greater'
                ).pvalue
            )
            self.fraud_detected = self.probability_honest < self.fraud_threshold
