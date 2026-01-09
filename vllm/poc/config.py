from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PoCState(Enum):
    IDLE = "IDLE"
    GENERATING = "GENERATING"
    VALIDATING = "VALIDATING"
    STOPPED = "STOPPED"


@dataclass
class PoCConfig:
    block_hash: str
    block_height: int
    public_key: str
    r_target: float
    fraud_threshold: float = 0.01
    node_id: int = 0
    node_count: int = 1
    batch_size: int = 32
    seq_len: int = 256
    callback_url: Optional[str] = None
