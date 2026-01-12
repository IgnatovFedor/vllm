"""PoC (Proof of Compute) API routes for vLLM server."""
import asyncio
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from vllm.logger import init_logger
from vllm.poc.config import PoCState, PoCConfig

logger = init_logger(__name__)

router = APIRouter(prefix="/api/v1/pow", tags=["PoC"])

# Module-level state for PoC tasks (per-app, keyed by id(app))
_poc_tasks: Dict[int, Dict[str, Any]] = {}

# Generate endpoint state
ConfigKey = Tuple[str, str, int]  # (block_hash, public_key, block_height)


@dataclass
class GenerateGroup:
    config: PoCConfig
    batch_size: int = 32
    nonce_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    results: Dict[str, List[Dict]] = field(default_factory=dict)
    callbacks: Dict[str, str] = field(default_factory=dict)
    last_callback: float = field(default_factory=time.time)
    # Stats tracking
    start_time: float = field(default_factory=time.time)
    total_processed: int = 0
    total_valid: int = 0
    # Blocking wait support
    pending_counts: Dict[str, int] = field(default_factory=dict)  # req_id -> remaining count
    completion_events: Dict[str, asyncio.Event] = field(default_factory=dict)  # req_id -> event
    # Vector return support
    return_vectors_flags: Dict[str, bool] = field(default_factory=dict)  # req_id -> wants vectors


_generate_groups: Dict[ConfigKey, GenerateGroup] = {}
_generate_worker: Optional[asyncio.Task] = None
_generate_lock: Optional[asyncio.Lock] = None


def _get_generate_lock() -> asyncio.Lock:
    global _generate_lock
    if _generate_lock is None:
        _generate_lock = asyncio.Lock()
    return _generate_lock


class PoCInitRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    r_target: float
    fraud_threshold: float = 0.01
    node_id: int = -1
    node_count: int = -1
    batch_size: int = 32
    seq_len: int = 256
    callback_url: Optional[str] = None


class PoCStatusResponse(BaseModel):
    state: str
    valid_nonces: List[int]
    valid_distances: List[float]
    total_checked: int
    total_valid: int
    elapsed_seconds: float
    rate_per_second: float


class PoCValidateRequest(BaseModel):
    """Request to validate nonces - accepts full ProofBatch format."""
    public_key: str
    block_hash: str
    block_height: int
    nonces: List[int]
    dist: List[float]
    node_id: int


class PoCGenerateRequest(BaseModel):
    """Request to generate distances for specific nonces."""
    block_hash: str
    block_height: int
    public_key: str
    r_target: float
    nonces: List[int]
    node_id: int = 0
    seq_len: int = 256
    batch_size: int = 32
    callback_url: Optional[str] = None
    wait: bool = False  # If True, block until all nonces are processed
    return_vectors: bool = False  # If True, return output vectors (requires wait=True)


async def get_engine_client(request: Request):
    """Get engine client from request app state."""
    engine_client = getattr(request.app.state, 'engine_client', None)
    if engine_client is None:
        raise HTTPException(status_code=503, detail="Engine not available")
    return engine_client


async def check_poc_enabled(request: Request):
    """Check if PoC is enabled."""
    poc_enabled = getattr(request.app.state, 'poc_enabled', False)
    if not poc_enabled:
        raise HTTPException(status_code=503, detail="PoC not enabled")


async def _cancel_poc_tasks(app_id: int):
    """Cancel running PoC tasks for an app."""
    tasks = _poc_tasks.pop(app_id, None)
    if tasks:
        if tasks.get("stop_event"):
            tasks["stop_event"].set()
        if tasks.get("gen_task"):
            tasks["gen_task"].cancel()
            try:
                await tasks["gen_task"]
            except asyncio.CancelledError:
                pass
        if tasks.get("send_task"):
            tasks["send_task"].cancel()
            try:
                await tasks["send_task"]
            except asyncio.CancelledError:
                pass


async def _cleanup_generate_groups():
    """Clean up /generate queues and worker."""
    global _generate_groups, _generate_worker
    
    # Clear all generate groups
    async with _get_generate_lock():
        _generate_groups.clear()
    
    # Cancel worker task
    if _generate_worker is not None and not _generate_worker.done():
        _generate_worker.cancel()
        try:
            await _generate_worker
        except asyncio.CancelledError:
            pass
        _generate_worker = None
    
    logger.info("Generate queues cleaned up")


async def _generation_loop(
    engine_client,
    batch_queue: asyncio.Queue,
    r_target: float,
):
    """Runs batches continuously, puts valid results in queue for callback sender."""
    total_checked = 0
    total_valid = 0
    batch_count = 0
    start_time = time.time()
    last_report_time = start_time
    
    logger.info(f"PoC generation started (r_target={r_target})")
    
    try:
        while True:
            result = await engine_client.poc_request("run_batch_with_state", {})
            
            if not result.get("should_continue", False):
                break
            
            batch_count += 1
            batch_nonces = len(result.get("nonces", []))
            batch_valid = len(result.get("valid_nonces", []))
            total_checked += batch_nonces
            total_valid += batch_valid
            
            # Log progress every 5 seconds
            current_time = time.time()
            if current_time - last_report_time >= 5.0:
                elapsed_min = (current_time - start_time) / 60
                valid_pct = 100 * total_valid / total_checked if total_checked > 0 else 0
                valid_rate = total_valid / elapsed_min if elapsed_min > 0 else 0
                raw_rate = total_checked / elapsed_min if elapsed_min > 0 else 0
                logger.info(f"Generated: {total_valid} / {total_checked} "
                           f"({valid_pct:.1f} from 100) Time: {elapsed_min:.2f}min "
                           f"({valid_rate:.1f} valid/min, {raw_rate:.0f} raw/min)")
                last_report_time = current_time
            
            # Put valid batch in queue for sender (non-blocking)
            if result.get("valid_nonces"):
                await batch_queue.put({
                    "public_key": result["public_key"],
                    "block_hash": result["block_hash"],
                    "block_height": result["block_height"],
                    "nonces": result["valid_nonces"],
                    "dist": result["valid_distances"],
                    "node_id": result["node_id"],
                    "r_target": r_target,
                })
    except asyncio.CancelledError:
        elapsed_min = (time.time() - start_time) / 60
        valid_pct = 100 * total_valid / total_checked if total_checked > 0 else 0
        valid_rate = total_valid / elapsed_min if elapsed_min > 0 else 0
        logger.info(f"PoC stopped: {total_valid} / {total_checked} ({valid_pct:.1f} from 100) "
                   f"in {elapsed_min:.2f}min ({valid_rate:.1f} valid/min)")


async def _callback_sender_loop(
    batch_queue: asyncio.Queue,
    callback_url: str,
    stop_event: asyncio.Event,
):
    """Sends batches from queue to callback URL."""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            try:
                # Wait for batch with timeout to check stop_event
                batch = await asyncio.wait_for(
                    batch_queue.get(),
                    timeout=1.0
                )
                try:
                    await session.post(
                        f"{callback_url}/generated",
                        json=batch,
                        timeout=aiohttp.ClientTimeout(total=10)
                    )
                    logger.debug(f"Callback sent to {callback_url}/generated")
                except Exception as e:
                    logger.warning(f"Callback failed: {e}")
            except asyncio.TimeoutError:
                continue  # Check stop_event
            except asyncio.CancelledError:
                break


async def _start_generation_tasks(
    request: Request,
    engine_client,
    callback_url: Optional[str],
    r_target: float,
):
    """Start generation loop and optional callback sender tasks."""
    app_id = id(request.app)
    
    # Cancel existing tasks
    await _cancel_poc_tasks(app_id)
    
    # Create queue and stop event
    batch_queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()
    
    # Start generation loop
    gen_task = asyncio.create_task(
        _generation_loop(engine_client, batch_queue, r_target)
    )
    
    # Start callback sender if URL provided
    send_task = None
    if callback_url:
        send_task = asyncio.create_task(
            _callback_sender_loop(batch_queue, callback_url, stop_event)
        )
    
    # Store for cleanup
    _poc_tasks[app_id] = {
        "gen_task": gen_task,
        "send_task": send_task,
        "stop_event": stop_event,
        "queue": batch_queue,
    }


@router.post("/init")
async def init_round(request: Request, body: PoCInitRequest) -> dict:
    """Initialize PoC round without starting generation."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    result = await engine_client.poc_request("init", body.model_dump())
    return {"status": "OK", "pow_status": result.get("pow_status", {})}


@router.post("/init/generate")
async def init_generate(request: Request, body: PoCInitRequest) -> dict:
    """Initialize PoC round and start generating."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    if body.node_id == -1 or body.node_count == -1:
        raise HTTPException(
            status_code=400,
            detail="Node ID and node count must be set"
        )
    
    # Initialize and start generating
    await engine_client.poc_request("init", body.model_dump())
    result = await engine_client.poc_request("start_generate", {})
    
    # Start background tasks
    await _start_generation_tasks(request, engine_client, body.callback_url, body.r_target)
    
    return {"status": "OK", "pow_status": result.get("pow_status", {})}


@router.post("/init/validate")
async def init_validate(request: Request, body: PoCInitRequest) -> dict:
    """Initialize PoC round and start validating."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    app_id = id(request.app)
    
    # Cancel any generation tasks
    await _cancel_poc_tasks(app_id)
    
    # Store callback URL for validation results (per-app)
    # Also init stop_event/queue to match expected structure
    _poc_tasks[app_id] = {
        "callback_url": body.callback_url,
        "stop_event": asyncio.Event(),
        "queue": asyncio.Queue(),
    }
    
    # Initialize and start validating
    await engine_client.poc_request("init", body.model_dump())
    result = await engine_client.poc_request("start_validate", {})
    
    return {"status": "OK", "pow_status": result.get("pow_status", {})}


@router.post("/phase/generate")
async def start_generate(request: Request) -> dict:
    """Switch to generate mode."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Check if initialized: after `/init`, PoC stays in IDLE but has config.
    status = await engine_client.poc_request("status", {})
    r_target = status.get("r_target")
    if r_target is None:
        raise HTTPException(status_code=400, detail="PoC not initialized (missing config)")
    
    result = await engine_client.poc_request("start_generate", {})
    
    # Start background tasks (no callback URL since round already initialized)
    await _start_generation_tasks(request, engine_client, None, r_target)
    
    return {"status": "OK", "pow_status": result.get("pow_status", {})}


@router.post("/phase/validate")
async def start_validate(request: Request) -> dict:
    """Switch to validate mode."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Check if initialized
    status = await engine_client.poc_request("status", {})
    if status.get("state") == PoCState.IDLE.value:
        raise HTTPException(status_code=400, detail="PoC not initialized")
    
    # Cancel generation tasks
    await _cancel_poc_tasks(id(request.app))
    
    result = await engine_client.poc_request("start_validate", {})
    
    return {"status": "OK", "pow_status": result.get("pow_status", {})}


@router.post("/stop")
async def stop_round(request: Request) -> dict:
    """Stop current PoC round."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Cancel all PoC tasks and queues
    await _cancel_poc_tasks(id(request.app))
    await _cleanup_generate_groups()
    
    result = await engine_client.poc_request("stop", {})
    
    return {"status": "OK", "pow_status": result.get("pow_status", {})}


@router.post("/batch")
async def run_one_batch(request: Request) -> dict:
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)

    status = await engine_client.poc_request("status", {})
    if status.get("state") != PoCState.GENERATING.value:
        raise HTTPException(
            status_code=400,
            detail="PoC must be in GENERATING state to run a batch.",
        )

    return await engine_client.poc_request("run_batch_with_state", {})


@router.get("/status", response_model=PoCStatusResponse)
async def get_status(request: Request) -> PoCStatusResponse:
    """Get current PoC status."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    status = await engine_client.poc_request("status", {})
    return PoCStatusResponse(**status)


@router.post("/validate")
async def validate_nonces(request: Request, body: PoCValidateRequest) -> dict:
    """Validate submitted nonces by recomputing distances.
    
    Accepts full ProofBatch format (matching original API).
    Results sent to callback_url/validated if configured.
    """
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Check that we have a round configured
    status = await engine_client.poc_request("status", {})
    if status.get("state") == PoCState.IDLE.value:
        raise HTTPException(status_code=400, detail="No round configured")
    
    # Validate and get results
    result = await engine_client.poc_request("queue_validation", {
        "public_key": body.public_key,
        "block_hash": body.block_hash,
        "nonces": body.nonces,
        "dist": body.dist,
    })
    
    # Send callback if URL configured (per-app)
    app_id = id(request.app)
    callback_url = _poc_tasks.get(app_id, {}).get("callback_url")
    if callback_url:
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{callback_url}/validated",
                    json=result,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Validation callback failed: {resp.status}")
        except Exception as e:
            logger.warning(f"Validation callback error: {e}")
    
    return {
        "status": "OK", 
        "fraud_detected": result.get("fraud_detected", False),
        "computed_distances": result.get("computed_distances", []),
    }


async def _send_generate_callbacks(group: GenerateGroup):
    """Send accumulated results to callback URLs."""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        for req_id, results in list(group.results.items()):
            if not results:
                continue
            
            callback_url = group.callbacks.get(req_id)
            if not callback_url:
                continue
            
            # Extract nonces and distances (matching /init/generate format)
            nonces = [r["nonce"] for r in results]
            distances = [r["distance"] for r in results if r["distance"] is not None]
            
            payload = {
                "request_id": req_id,
                "block_hash": group.config.block_hash,
                "block_height": group.config.block_height,
                "public_key": group.config.public_key,
                "r_target": group.config.r_target,
                "nonces": nonces,
                "dist": distances,
            }
            
            logger.info(f"Sending callback: {len(nonces)} nonces to {callback_url}")
            
            try:
                await session.post(
                    f"{callback_url}/generated",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                )
                group.results[req_id] = []
            except Exception as e:
                logger.warning(f"Generate callback failed for {req_id}: {e}")


async def _generate_worker_loop(engine_client):
    """Process nonces from generate groups."""
    global _generate_groups
    
    logger.info("Generate worker started")
    
    try:
        while _generate_groups:
            # Collect batch under lock
            batch_info = None
            async with _get_generate_lock():
                for key, group in list(_generate_groups.items()):
                    batch_nonces = []
                    batch_req_ids = []
                    batch_size = group.batch_size
                    
                    while len(batch_nonces) < batch_size:
                        try:
                            nonce, req_id = group.nonce_queue.get_nowait()
                            batch_nonces.append(nonce)
                            batch_req_ids.append(req_id)
                        except asyncio.QueueEmpty:
                            break
                    
                    if batch_nonces:
                        batch_info = (key, group.config, batch_nonces, batch_req_ids)
                        break
            
            # Process batch without lock
            if batch_info:
                key, config, batch_nonces, batch_req_ids = batch_info
                try:
                    # Check if any request in batch wants vectors
                    needs_vectors = False
                    async with _get_generate_lock():
                        if key in _generate_groups:
                            group = _generate_groups[key]
                            needs_vectors = any(
                                group.return_vectors_flags.get(rid, False) 
                                for rid in set(batch_req_ids)
                            )
                    
                    result = await engine_client.poc_request("generate_for_nonces", {
                        "block_hash": config.block_hash,
                        "block_height": config.block_height,
                        "public_key": config.public_key,
                        "r_target": config.r_target,
                        "seq_len": config.seq_len,
                        "nonces": batch_nonces,
                        "return_vectors": needs_vectors,
                    })
                    
                    # Store results
                    async with _get_generate_lock():
                        if key in _generate_groups:
                            group = _generate_groups[key]
                            distances = result.get("distances", [])
                            vectors = result.get("vectors", [])
                            r_target = config.r_target
                            valid_count = 0
                            
                            # Count processed nonces per request for pending tracking
                            processed_per_req: Dict[str, int] = {}
                            
                            for i, req_id in enumerate(batch_req_ids):
                                processed_per_req[req_id] = processed_per_req.get(req_id, 0) + 1
                                dist = distances[i] if i < len(distances) else None
                                wants_vectors = group.return_vectors_flags.get(req_id, False)
                                
                                # Store ALL results if vectors requested, otherwise only valid
                                if wants_vectors or (dist is not None and dist < r_target):
                                    if req_id not in group.results:
                                        group.results[req_id] = []
                                    entry = {
                                        "nonce": batch_nonces[i],
                                        "distance": dist,
                                    }
                                    if wants_vectors and vectors:
                                        entry["vector"] = vectors[i] if i < len(vectors) else None
                                    group.results[req_id].append(entry)
                                    if dist is not None and dist < r_target:
                                        valid_count += 1
                            
                            # Update pending counts and fire completion events
                            for req_id, count in processed_per_req.items():
                                if req_id in group.pending_counts:
                                    group.pending_counts[req_id] -= count
                                    if group.pending_counts[req_id] <= 0:
                                        # All nonces processed, fire completion event
                                        event = group.completion_events.get(req_id)
                                        if event:
                                            event.set()
                            
                            # Update stats
                            group.total_processed += len(batch_nonces)
                            group.total_valid += valid_count
                            elapsed = (time.time() - group.start_time) / 60.0
                            valid_rate = group.total_valid / group.total_processed * 100 if group.total_processed else 0
                            valid_per_min = group.total_valid / elapsed if elapsed > 0 else 0
                            raw_per_min = group.total_processed / elapsed if elapsed > 0 else 0
                            
                            logger.info(
                                f"Generated: {group.total_valid} / {group.total_processed} "
                                f"({valid_rate:.1f}%) Time: {elapsed:.2f}min "
                                f"({valid_per_min:.1f} valid/min, {raw_per_min:.0f} raw/min)"
                            )
                except Exception as e:
                    logger.error(f"Generate batch failed: {e}\n{traceback.format_exc()}")
            
            # Check callbacks and cleanup under lock
            async with _get_generate_lock():
                for key, group in list(_generate_groups.items()):
                    has_results = any(r for r in group.results.values())
                    time_elapsed = time.time() - group.last_callback >= 10.0
                    queue_done = group.nonce_queue.empty()
                    
                    if has_results and (time_elapsed or queue_done):
                        await _send_generate_callbacks(group)
                        group.last_callback = time.time()
                    
                    if queue_done and not any(r for r in group.results.values()):
                        del _generate_groups[key]
            
            if not batch_info:
                await asyncio.sleep(0.01)
    except Exception as e:
        logger.error(f"Generate worker error: {e}")
    finally:
        # Cleanup hooks and release memory when worker stops
        try:
            await engine_client.poc_request("teardown_generate_hooks", {})
        except Exception:
            pass
        logger.info("Generate worker stopped")


async def _start_generate_worker(engine_client):
    """Start the generate worker if not running."""
    global _generate_worker
    
    if _generate_worker is None or _generate_worker.done():
        _generate_worker = asyncio.create_task(
            _generate_worker_loop(engine_client)
        )


@router.post("/generate")
async def generate_nonces(request: Request, body: PoCGenerateRequest) -> dict:
    """Generate distances for specific nonces.
    
    Groups requests by (block_hash, public_key, block_height) for efficient batching.
    Results sent to callback_url/generated every ~10s.
    
    If wait=True, blocks until all nonces are processed and returns results directly.
    If return_vectors=True (requires wait=True), also returns the output vectors.
    """
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    status = await engine_client.poc_request("status", {})
    if status.get("state") in [PoCState.GENERATING.value, PoCState.VALIDATING.value]:
        raise HTTPException(
            status_code=409,
            detail="Busy with /init/generate or /init/validate"
        )
    
    if body.return_vectors and not body.wait:
        raise HTTPException(
            status_code=400,
            detail="return_vectors requires wait=True"
        )
    
    key: ConfigKey = (body.block_hash, body.public_key, body.block_height)
    req_id = str(uuid.uuid4())
    completion_event = None
    
    async with _get_generate_lock():
        if key not in _generate_groups:
            config = PoCConfig(
                block_hash=body.block_hash,
                block_height=body.block_height,
                public_key=body.public_key,
                r_target=body.r_target,
                seq_len=body.seq_len,
                node_id=body.node_id,
            )
            _generate_groups[key] = GenerateGroup(config=config, batch_size=body.batch_size)
        
        group = _generate_groups[key]
        group.results[req_id] = []
        if body.callback_url:
            group.callbacks[req_id] = body.callback_url
        
        # Track if this request wants vectors
        if body.return_vectors:
            group.return_vectors_flags[req_id] = True
        
        # Set up blocking wait if requested
        if body.wait:
            completion_event = asyncio.Event()
            group.completion_events[req_id] = completion_event
            group.pending_counts[req_id] = len(body.nonces)
        
        for nonce in body.nonces:
            await group.nonce_queue.put((nonce, req_id))
    
    await _start_generate_worker(engine_client)
    
    # If wait=True, block until all nonces are processed
    if body.wait and completion_event:
        await completion_event.wait()
        
        # Collect results
        async with _get_generate_lock():
            if key in _generate_groups:
                group = _generate_groups[key]
                results = group.results.pop(req_id, [])
                group.completion_events.pop(req_id, None)
                group.pending_counts.pop(req_id, None)
                group.return_vectors_flags.pop(req_id, None)
            else:
                results = []
        
        nonces = [r["nonce"] for r in results]
        distances = [r["distance"] for r in results]
        
        response = {
            "status": "completed",
            "request_id": req_id,
            "valid_nonces": nonces,
            "valid_distances": distances,
            "total_valid": len(nonces),
        }
        
        if body.return_vectors:
            response["vectors"] = [r.get("vector") for r in results]
        
        return response
    
    return {
        "status": "queued",
        "request_id": req_id,
        "queued_count": len(body.nonces),
    }
