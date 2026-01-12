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
    generate_haar_orthogonal_matrices,
)

# Number of dimensions to pick for distance computation
POC_PICK_K_DIMS = 12


def _create_prefill_attn_metadata(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    attn_backend,
    dtype: torch.dtype,
):
    """Create prefill attention metadata for the given backend.
    
    Uses PAD_SLOT_ID for all slots to skip KV cache writes.
    """
    num_tokens = batch_size * seq_len
    seq_lens = [seq_len] * batch_size
    
    seq_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    seq_start_loc[1:] = torch.cumsum(
        torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0
    )
    
    backend_name = attn_backend.get_name()
    
    if backend_name == "XFORMERS":
        from vllm.v1.attention.backends.xformers import XFormersAttentionMetadata
        return XFormersAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=seq_len,
            query_start_loc=seq_start_loc,
            max_seq_len=seq_len,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            block_table=torch.empty((batch_size, 0), dtype=torch.int32, device=device),
            slot_mapping=torch.full((num_tokens,), PAD_SLOT_ID, dtype=torch.long, device=device),
            num_prefills=batch_size,
            num_prefill_tokens=num_tokens,
            num_decodes=0,
            num_decode_tokens=0,
        )
    elif backend_name == "FLASHINFER":
        from vllm.v1.attention.backends.flashinfer import FlashInferMetadata
        return FlashInferMetadata(
            num_actual_tokens=num_tokens,
            q_data_type=dtype,
            slot_mapping=torch.full((num_tokens,), PAD_SLOT_ID, dtype=torch.long, device=device),
            max_q_len=seq_len,
            max_q_len_prefill=seq_len,
            max_seq_len=seq_len,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            block_table_tensor=torch.empty((batch_size, 0), dtype=torch.int32, device=device),
            prefill_use_trtllm=False,
            decode_use_trtllm=False,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=batch_size,
            num_prefill_tokens=num_tokens,
            use_cascade=False,
        )
    else:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
        return FlashAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=seq_len,
            query_start_loc=seq_start_loc,
            max_seq_len=seq_len,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            block_table=torch.empty((batch_size, 0), dtype=torch.int32, device=device),
            slot_mapping=torch.full((num_tokens,), PAD_SLOT_ID, dtype=torch.long, device=device),
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
    r_target: float,
    vllm_config,  # Kept for API compatibility
    return_vectors: bool = False,
) -> Optional[Dict[str, Any]]:
    """Execute PoC forward pass on a worker.
    
    Mimics /chat/completion TP synchronization:
    - TP rank0 broadcasts PoC metadata
    - Non-driver ranks block until broadcast received
    - All ranks enter forward together (NCCL ops align)
    """
    device = worker.device
    dtype = worker.model_runner.model_config.dtype
    model = worker.model_runner.model
    worker_vllm_config = worker.vllm_config
    
    tp_group = get_tp_group()
    is_tp_driver = tp_group.rank_in_group == 0
    
    # =========================================================================
    # TP SYNC: Rendezvous + CPU-only gate (no NCCL)
    # 
    # 1. CPU barrier ensures all TP ranks have ENTERED execute_poc_forward
    #    before driver broadcasts (prevents driver racing ahead)
    # 2. Driver broadcasts Python values via CPU group (Gloo), non-drivers block.
    # 
    # This mimics /chat/completion semantics WITHOUT adding NCCL collectives
    # that could get out-of-order with model-forward NCCL.
    # =========================================================================
    if tp_group.world_size > 1:
        # Rendezvous: ensure all TP ranks have entered before broadcast
        dist.barrier(group=tp_group.cpu_group)
        
        if is_tp_driver:
            # Driver: broadcast PoC metadata (Python values only - uses CPU group)
            broadcast_tensor_dict({
                "poc_go": True,  # signal
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "nonces": nonces,
                "return_vectors": return_vectors,
            }, src=0)
        else:
            # Non-driver: block here until driver broadcasts (like /chat/completion)
            broadcast_data = broadcast_tensor_dict(src=0)
            # Use broadcasted values (ensures all TP ranks have identical params)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            return_vectors = bool(broadcast_data["return_vectors"])
    
    batch_size = len(nonces)
    
    # Generate embeddings on first PP rank, receive intermediate tensors on others
    intermediate_tensors = None
    inputs_embeds = None
    
    pp_group = get_pp_group()
    
    if pp_group.is_first_rank:
        # Generate deterministic inputs on GPU (all TP ranks do this with same params)
        inputs_embeds = generate_inputs(
            block_hash, public_key, nonces,
            dim=hidden_size, seq_len=seq_len,
            device=device, dtype=dtype,
        )
    else:
        # Receive from previous PP rank
        intermediate_tensors = IntermediateTensors(
            pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
        )
    
    # Create attention metadata and positions
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Get attention backend from attn_groups
    if hasattr(worker.model_runner, 'attn_groups') and len(worker.model_runner.attn_groups) > 0:
        # Get the first attention group from the first kv_cache_group
        attn_backend = worker.model_runner.attn_groups[0][0].backend
    else:
        # Fallback: try to get from model layers directly
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.config import get_layers_from_vllm_config
        layers = get_layers_from_vllm_config(worker_vllm_config, AttentionLayerBase, None)
        if layers:
            # Get backend from first attention layer
            first_layer_name = list(layers.keys())[0]
            attn_backend = layers[first_layer_name].get_attn_backend()
        else:
            raise AttributeError("Cannot determine attention backend: attn_groups not initialized and no layers found")
    
    # Create attention metadata - v1 models expect a dict mapping layer names to metadata
    single_attn_metadata = _create_prefill_attn_metadata(batch_size, seq_len, device, attn_backend, dtype)
    
    # Get all attention layer names from the model
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.config import get_layers_from_vllm_config
    attention_layers = get_layers_from_vllm_config(worker_vllm_config, AttentionLayerBase, None)
    
    # Create dict mapping each layer name to the same metadata
    attn_metadata = {layer_name: single_attn_metadata for layer_name in attention_layers.keys()}
    
    # =========================================================================
    # TP SYNC: Pre-forward rendezvous (after PP recv, before model forward)
    # 
    # Ensures all TP ranks in this PP stage enter model forward together.
    # For PP stage 0: all ranks finished generate_inputs
    # For PP stage >0: all ranks finished recv_tensor_dict
    # =========================================================================
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
    
    # Sync GPU before forward to ensure all CUDA ops complete
    torch.cuda.synchronize()
    
    # Forward pass - all TP ranks now enter together
    # Pass num_tokens to set_forward_context for proper initialization
    num_tokens = batch_size * seq_len
    
    with set_forward_context(attn_metadata, worker_vllm_config, num_tokens=num_tokens):
        hidden_states = model(
            input_ids=None,
            positions=positions.flatten(),
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds.view(-1, hidden_size) if inputs_embeds is not None else None,
        )
        
        # Sync inside the forward context to catch errors early
        torch.cuda.synchronize()
    
    # PP: send to next rank if not last
    if not pp_group.is_last_rank:
        if isinstance(hidden_states, IntermediateTensors):
            pp_group.send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group()
            )
        return None
    
    # Extract last token hidden state
    # Ensure hidden_states is contiguous and has correct shape
    num_tokens = batch_size * seq_len
    if hidden_states.shape[0] != num_tokens:
        raise ValueError(
            f"Shape mismatch: hidden_states.shape[0]={hidden_states.shape[0]}, "
            f"expected {num_tokens} (batch_size={batch_size} * seq_len={seq_len})"
        )
    
    # Reshape to (batch_size, seq_len, hidden_size)
    hidden_states = hidden_states.view(batch_size, seq_len, -1)
    
    # Extract last token: (batch_size, hidden_size)
    # Clone to ensure we have a valid tensor with its own memory
    last_hidden = hidden_states[:, -1, :].clone().float()
    
    # Ensure last_hidden is contiguous before norm operation
    if not last_hidden.is_contiguous():
        last_hidden = last_hidden.contiguous()
    
    # Normalize to unit sphere
    last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Per-nonce k-dim pick + Haar rotation
    indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, POC_PICK_K_DIMS, device)
    xk = torch.gather(last_hidden, 1, indices)
    
    Q = generate_haar_orthogonal_matrices(block_hash, public_key, nonces, POC_PICK_K_DIMS, device, dtype=xk.dtype)
    yk = torch.bmm(Q, xk.unsqueeze(-1)).squeeze(-1)
    
    # Target in k-dim space (per-nonce)
    target = generate_target(block_hash, public_key, POC_PICK_K_DIMS, device)
    
    # Normalize and compute distances
    yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)
    target = target / (target.norm(dim=-1, keepdim=True) + 1e-8)
    distances = (yk - target).norm(dim=-1)
    
    result = {
        "nonces": nonces,
        "distances": distances.cpu().tolist(),
    }
    if return_vectors:
        result["vectors"] = yk.cpu().tolist()
    
    return result
