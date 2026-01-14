"""PoC model runner - simplified forward pass.

This mimics vLLM's /chat/completion TP synchronization:
- TP rank0 (driver) broadcasts metadata to all TP workers
- Non-driver TP workers block until they receive the broadcast
- All TP ranks then enter model forward together (NCCL collectives align)
"""
import torch
import torch.distributed as dist
from typing import List, Optional, Dict, Any

from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors

from .gpu_random import (
    generate_inputs,
    generate_target,
    random_pick_indices,
    apply_haar_rotation,
)

# Default k_dim (can be overridden per-request)
DEFAULT_K_DIM = 12


def _create_prefill_attn_metadata(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    attn_backend,
):
    """Create prefill attention metadata for the v1 attention backend.

    Uses PAD_SLOT_ID for all slots to skip KV cache writes.
    This creates v1-style FlashAttentionMetadata.
    """
    num_tokens = batch_size * seq_len

    # query_start_loc: cumulative query lengths [0, seq_len, 2*seq_len, ...]
    query_start_loc = torch.arange(
        0, num_tokens + 1, seq_len, dtype=torch.int32, device=device
    )

    # seq_lens: tensor of sequence lengths
    seq_lens_tensor = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    # slot_mapping: all PAD_SLOT_ID to skip KV cache writes
    slot_mapping = torch.full((num_tokens,), PAD_SLOT_ID, dtype=torch.long, device=device)

    # block_table: empty since we're not using KV cache
    block_table = torch.empty((batch_size, 0), dtype=torch.int32, device=device)

    # Import v1 FlashAttentionMetadata
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    return FlashAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=seq_len,
        query_start_loc=query_start_loc,
        max_seq_len=seq_len,
        seq_lens=seq_lens_tensor,
        block_table=block_table,
        slot_mapping=slot_mapping,
        # Cascade attention disabled for PoC
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )


@torch.inference_mode()
def execute_poc_forward(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    hidden_size: int,
    k_dim: int = DEFAULT_K_DIM,
) -> Optional[Dict[str, Any]]:
    """Execute PoC forward pass on a worker.
    
    Mimics /chat/completion TP synchronization:
    - TP rank0 broadcasts PoC metadata
    - Non-driver ranks block until broadcast received
    - All ranks enter forward together (NCCL ops align)
    
    Returns:
        Dict with nonces and vectors (FP16 numpy arrays for encoding).
        Returns None for non-last PP ranks.
    """
    device = worker.device
    dtype = worker.model_runner.model_config.dtype
    model = worker.model_runner.model
    worker_vllm_config = worker.vllm_config
    
    tp_group = get_tp_group()
    is_tp_driver = tp_group.rank_in_group == 0
    
    # =========================================================================
    # TP SYNC: Rendezvous + CPU-only gate (no NCCL)
    # =========================================================================
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
        
        if is_tp_driver:
            broadcast_tensor_dict({
                "poc_go": True,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "nonces": nonces,
                "k_dim": k_dim,
            }, src=0)
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            k_dim = int(broadcast_data["k_dim"])
    
    batch_size = len(nonces)
    
    # Generate embeddings on first PP rank, receive intermediate tensors on others
    intermediate_tensors = None
    inputs_embeds = None
    
    pp_group = get_pp_group()
    
    if pp_group.is_first_rank:
        inputs_embeds = generate_inputs(
            block_hash, public_key, nonces,
            dim=hidden_size, seq_len=seq_len,
            device=device, dtype=dtype,
        )
    else:
        intermediate_tensors = IntermediateTensors(
            pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
        )
    
    # Create positions tensor
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # For v1 architecture, we pass attn_metadata=None like profile runs do.
    # This avoids the complexity of building per-layer attention metadata dictionaries.
    # The model will run without KV caching (which is what we want for PoC).
    model_runner = worker.model_runner
    if hasattr(model_runner, 'attn_groups'):
        # v1 architecture - use None like profile runs
        attn_metadata = None
    else:
        # v0 architecture - create attention metadata
        attn_backend = model_runner.attn_backend
        attn_metadata = _create_prefill_attn_metadata(batch_size, seq_len, device, attn_backend)
    
    # =========================================================================
    # TP SYNC: Pre-forward rendezvous
    # =========================================================================
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
    
    torch.cuda.synchronize()
    
    # Forward pass
    # Note: We pass a dummy input_ids tensor even though inputs_embeds will be used.
    # This is because torch.compile expects input_ids to be a tensor based on how
    # the model was compiled during profile run.
    num_tokens = batch_size * seq_len
    dummy_input_ids = torch.zeros(num_tokens, dtype=torch.long, device=device)

    with set_forward_context(attn_metadata, worker_vllm_config):
        hidden_states = model(
            input_ids=dummy_input_ids,
            positions=positions.flatten(),
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds.view(-1, hidden_size) if inputs_embeds is not None else None,
        )
    
    # PP: send to next rank if not last
    if not pp_group.is_last_rank:
        if isinstance(hidden_states, IntermediateTensors):
            pp_group.send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group()
            )
        return None
    
    # Extract last token hidden state and compute in FP32
    hidden_states = hidden_states.view(batch_size, seq_len, -1)
    last_hidden = hidden_states[:, -1, :].float()
    
    # Normalize to unit sphere
    last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Per-nonce k-dim pick + Haar rotation (via Householder chain, no cuSOLVER)
    indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, k_dim, device)
    xk = torch.gather(last_hidden, 1, indices)
    yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)
    
    # Normalize output vectors
    yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Convert to FP16 for artifact encoding (compute was in FP32)
    vectors_f16 = yk.half().cpu().numpy()
    
    return {
        "nonces": nonces,
        "vectors": vectors_f16,  # FP16 numpy array, shape [batch_size, k_dim]
    }
