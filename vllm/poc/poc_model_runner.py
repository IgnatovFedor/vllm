"""PoC model runner - simplified forward pass.

This mimics vLLM's /chat/completion TP synchronization:
- TP rank0 (driver) broadcasts metadata to all TP workers
- Non-driver TP workers block until they receive the broadcast
- All TP ranks then enter model forward together (NCCL collectives align)
"""
from typing import List, Optional, Dict, Any

import torch
import torch.distributed as dist
from vllm.platforms import current_platform
from vllm.attention.selector import get_attn_backend
from vllm.logger import init_logger
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.poc.gpu_random import (
    generate_inputs,
    random_pick_indices,
    apply_haar_rotation,
)
from vllm.poc.layer_hooks import LayerHouseholderHook, poc_forward_context
from vllm.sequence import IntermediateTensors

# Default k_dim (can be overridden per-request)
DEFAULT_K_DIM = 12
logger = init_logger(__name__)


def _ensure_layer_hooks(worker, block_hash: str, hidden_size: int) -> None:
    """Ensure layer hooks are installed on the worker for the given block_hash.
    
    Caches hooks on worker._poc_layer_hooks. If block_hash changes, detaches
    old hooks and installs new ones (per-round transform changes).
    """
    model = worker.model_runner.model
    device = worker.device
    
    existing_hook = getattr(worker, '_poc_layer_hooks', None)
    
    if existing_hook is not None:
        if existing_hook.block_hash == block_hash:
            return
        existing_hook.detach()
    
    hook = LayerHouseholderHook(model, block_hash, device, hidden_size)
    worker._poc_layer_hooks = hook


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
    model_config = worker.model_runner.model_config

    platform_name = current_platform.device_name
    device_capability = current_platform.get_device_capability()
    device_name = current_platform.get_device_name()

    # Get attention backend info
    head_size = model_config.get_head_size()
    kv_cache_dtype = getattr(worker_vllm_config.cache_config, 'kv_cache_dtype', None)
    block_size = getattr(worker_vllm_config.cache_config, 'block_size', 16)

    attn_backend_cls = get_attn_backend(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
    )
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

    actual_backend_info = "N/A"
    if hasattr(model_runner, 'attn_groups'):
        # v1 architecture
        if len(model_runner.attn_groups) > 0 and len(model_runner.attn_groups[0]) > 0:
            # Get first attention group's backend
            first_group = model_runner.attn_groups[0][0]
            if hasattr(first_group, 'backend'):
                actual_backend_info = first_group.backend.__class__.__name__
    elif hasattr(model_runner, 'attn_backend'):
        # v0 architecture
        actual_backend_info = model_runner.attn_backend.__class__.__name__

    logger.info(
        f"[TP Rank {tp_group.rank_in_group}/{tp_group.world_size}] "
        "=" * 80 + "\n"
        "HARDWARE & ATTENTION BACKEND INFO:\n"
        f"  Platform: {platform_name}\n"
        f"  Device: {device_name}\n"
        f"  Compute Capability: {device_capability}\n"
        f"  Dtype: {dtype}\n"
        f"  Head Size: {head_size}\n"
        f"  Block Size: {block_size}\n"
        f"  KV Cache Dtype: {kv_cache_dtype}\n"
        f"  Selected Attention Backend Class: {attn_backend_cls.__name__}\n"
        f"  Selected Attention Backend Module: {attn_backend_cls.__module__}\n"
        f"  Actual Model Backend: {actual_backend_info}\n"
        + "=" * 80
    )

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
    # Ensure layer hooks are installed for this block_hash (lazy + cached)
    # _ensure_layer_hooks(worker, block_hash, hidden_size)

    # Forward pass with PoC context (activates layer hook transformations)
    with set_forward_context(attn_metadata, worker_vllm_config):
        with poc_forward_context():
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

    # Detach and convert last_hidden for artifact/debug export
    last_hidden_f16 = last_hidden.half().cpu().numpy()
    
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
        "vectors": vectors_f16,       # FP16 numpy array, shape [batch_size, k_dim]
        "last_hidden": last_hidden_f16,  # FP16 numpy array, shape [batch_size, hidden_size]
    }
