"""PoC (Proof of Compute) API routes for vLLM server.

Implements artifact-based PoC protocol per production-phase-1.md.
All PoC state (generation loop, nonce counter, stats, generate queue) is managed here.
The engine only provides a single stateless operation: generate_artifacts.
"""
import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import aiohttp
import numpy as np
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from vllm.logger import init_logger
from .config import PoCState
from .data import (
    Artifact, decode_vector, fraud_test,
    DEFAULT_DIST_THRESHOLD, DEFAULT_P_MISMATCH, DEFAULT_FRAUD_THRESHOLD,
)

logger = init_logger(__name__)

router = APIRouter(prefix="/api/v1/pow", tags=["PoC"])

# Callback interval for /init/generate (seconds)
POC_CALLBACK_INTERVAL_SEC = float(os.environ.get("POC_CALLBACK_INTERVAL_SEC", "5"))

# Per-chunk timeout for /generate endpoint when engine is busy (seconds)
POC_GENERATE_CHUNK_TIMEOUT_SEC = float(os.environ.get("POC_GENERATE_CHUNK_TIMEOUT_SEC", "60"))

# Backoff sleep when PoC is skipped due to chat being busy (seconds)
POC_CHAT_BUSY_BACKOFF_SEC = 0.05

# Timeout for generate_artifacts RPC during coexistence (ms)
POC_RPC_TIMEOUT_MS = int(os.environ.get("POC_RPC_TIMEOUT_MS", "60000"))

# Result TTL for /generate queue results (seconds)
GENERATE_RESULT_TTL_SEC = float(os.environ.get("POC_GENERATE_RESULT_TTL_SEC", "300"))

# Module-level state: tracks active generation tasks per app
# Key: app_id, Value: dict with gen_task, send_task, stop_event, queue, config, stats
_poc_tasks: Dict[int, Dict[str, Any]] = {}


# =============================================================================
# Generate Queue Infrastructure
# =============================================================================

@dataclass
class GenerateJob:
    """A queued /generate request."""
    request_id: str
    engine_client: Any  # Reference to engine client
    app_id: int
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    nonces: List[int]
    seq_len: int
    k_dim: int
    batch_size: int
    validation_artifacts: Optional[Dict[int, str]] = None  # nonce -> vector_b64
    stat_test_dist_threshold: float = DEFAULT_DIST_THRESHOLD
    stat_test_p_mismatch: float = DEFAULT_P_MISMATCH
    stat_test_fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD
    callback_url: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class GenerateResult:
    """Result record for a queued /generate request."""
    status: str  # "queued", "running", "completed", "failed"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Module-level queue infrastructure
_generate_queue: asyncio.Queue = None  # Lazy init
_generate_results: Dict[str, GenerateResult] = {}
_generate_worker_task: Optional[asyncio.Task] = None
_generate_lock: asyncio.Lock = None  # Lazy init


def _ensure_queue_initialized():
    """Lazily initialize queue infrastructure."""
    global _generate_queue, _generate_lock
    if _generate_queue is None:
        _generate_queue = asyncio.Queue()
    if _generate_lock is None:
        _generate_lock = asyncio.Lock()


async def _ensure_worker_running(engine_client, app_id: int):
    """Ensure the generate worker is running."""
    global _generate_worker_task
    _ensure_queue_initialized()
    
    async with _generate_lock:
        if _generate_worker_task is None or _generate_worker_task.done():
            _generate_worker_task = asyncio.create_task(
                _generate_worker_loop(engine_client, app_id)
            )


async def _generate_worker_loop(engine_client, app_id: int):
    """Background worker that processes queued /generate jobs."""
    logger.info("Generate queue worker started")
    
    while True:
        try:
            # Get next job (blocks until available)
            job: GenerateJob = await _generate_queue.get()
            
            # Update status to running
            if job.request_id in _generate_results:
                _generate_results[job.request_id].status = "running"
            
            try:
                # Wait if /init/generate is active
                while _is_generation_active(job.app_id):
                    await asyncio.sleep(0.1)
                
                # Process the job
                result = await _process_generate_job(job)
                
                # Store result
                if job.request_id in _generate_results:
                    _generate_results[job.request_id].status = "completed"
                    _generate_results[job.request_id].completed_at = time.time()
                    _generate_results[job.request_id].result = result
                
                # Send callback if URL provided
                if job.callback_url:
                    await _send_generate_callback(job, result)
                    
            except Exception as e:
                logger.error(f"Generate job {job.request_id} failed: {e}", exc_info=True)
                if job.request_id in _generate_results:
                    _generate_results[job.request_id].status = "failed"
                    _generate_results[job.request_id].completed_at = time.time()
                    _generate_results[job.request_id].error = str(e)
            
            # Cleanup old results
            _cleanup_old_results()
            
        except asyncio.CancelledError:
            logger.info("Generate queue worker stopped")
            break
        except Exception as e:
            logger.error(f"Generate worker error: {e}", exc_info=True)
            await asyncio.sleep(1)  # Avoid tight loop on repeated errors


async def _process_generate_job(job: GenerateJob) -> Dict[str, Any]:
    """Process a single generate job (same logic as wait=true path)."""
    computed_artifacts = []
    
    for i in range(0, len(job.nonces), job.batch_size):
        chunk = job.nonces[i:i + job.batch_size]
        chunk_start_time = time.time()
        
        while True:
            # Wait if /init/generate became active
            while _is_generation_active(job.app_id):
                await asyncio.sleep(0.1)
            
            result = await job.engine_client.poc_request("generate_artifacts", {
                "nonces": chunk,
                "block_hash": job.block_hash,
                "public_key": job.public_key,
                "seq_len": job.seq_len,
                "k_dim": job.k_dim,
            })
            
            if not result.get("skipped"):
                computed_artifacts.extend(result.get("artifacts", []))
                break
            
            elapsed = time.time() - chunk_start_time
            if elapsed >= POC_GENERATE_CHUNK_TIMEOUT_SEC:
                raise RuntimeError(
                    f"Timeout waiting for engine: chunk {i//job.batch_size} "
                    f"timed out after {elapsed:.1f}s"
                )
            
            await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC)
    
    # If no validation, return artifacts
    if job.validation_artifacts is None:
        return {
            "status": "completed",
            "request_id": job.request_id,
            "artifacts": computed_artifacts,
            "encoding": {"dtype": "f16", "k_dim": job.k_dim, "endian": "le"},
        }
    
    # Validation mode
    n_mismatch = 0
    mismatch_nonces = []
    
    for artifact in computed_artifacts:
        nonce = artifact["nonce"]
        computed_b64 = artifact["vector_b64"]
        received_b64 = job.validation_artifacts.get(nonce)
        
        if received_b64:
            computed_vec = decode_vector(computed_b64)
            received_vec = decode_vector(received_b64)
            distance = np.linalg.norm(computed_vec - received_vec)
            
            if distance > job.stat_test_dist_threshold:
                n_mismatch += 1
                mismatch_nonces.append(nonce)
    
    n_total = len(job.nonces)
    p_value, fraud_detected = fraud_test(
        n_mismatch, n_total,
        job.stat_test_p_mismatch, job.stat_test_fraud_threshold
    )
    
    return {
        "status": "completed",
        "request_id": job.request_id,
        "n_total": n_total,
        "n_mismatch": n_mismatch,
        "mismatch_nonces": mismatch_nonces,
        "p_value": p_value,
        "fraud_detected": fraud_detected,
    }


async def _send_generate_callback(job: GenerateJob, result: Dict[str, Any]):
    """Send callback for completed generate job."""
    try:
        async with aiohttp.ClientSession() as session:
            if job.validation_artifacts is None:
                # Compute-only: POST to /generated
                payload = {
                    "request_id": job.request_id,
                    "block_hash": job.block_hash,
                    "block_height": job.block_height,
                    "public_key": job.public_key,
                    "node_id": job.node_id,
                    "artifacts": result.get("artifacts", []),
                    "encoding": result.get("encoding", {}),
                }
                await session.post(
                    f"{job.callback_url}/generated",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                )
            else:
                # Validation: POST to /validated
                payload = {
                    "request_id": job.request_id,
                    "block_hash": job.block_hash,
                    "block_height": job.block_height,
                    "public_key": job.public_key,
                    "node_id": job.node_id,
                    "n_total": result.get("n_total", 0),
                    "n_mismatch": result.get("n_mismatch", 0),
                    "mismatch_nonces": result.get("mismatch_nonces", []),
                    "p_value": result.get("p_value", 1.0),
                    "fraud_detected": result.get("fraud_detected", False),
                }
                await session.post(
                    f"{job.callback_url}/validated",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                )
    except Exception as e:
        logger.warning(f"Generate callback failed for {job.request_id}: {e}")


def _cleanup_old_results():
    """Remove results older than TTL."""
    now = time.time()
    expired = [
        rid for rid, rec in _generate_results.items()
        if (rec.completed_at and now - rec.completed_at > GENERATE_RESULT_TTL_SEC)
        or (not rec.completed_at and now - rec.created_at > GENERATE_RESULT_TTL_SEC * 2)
    ]
    for rid in expired:
        del _generate_results[rid]


# =============================================================================
# Request/Response Models
# =============================================================================

class PoCParamsModel(BaseModel):
    """Strict params for PoC requests."""
    model: str
    seq_len: int
    k_dim: int = 12


class PoCInitGenerateRequest(BaseModel):
    """Request for /init/generate endpoint."""
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    batch_size: int = 32
    params: PoCParamsModel
    url: Optional[str] = None


class ArtifactModel(BaseModel):
    """Single artifact for request/response."""
    nonce: int
    vector_b64: str


class EncodingModel(BaseModel):
    """Encoding metadata."""
    dtype: str = "f16"
    k_dim: int = 12
    endian: str = "le"


class ValidationModel(BaseModel):
    """Artifacts to validate against."""
    artifacts: List[ArtifactModel]


class StatTestModel(BaseModel):
    """Statistical test parameters."""
    dist_threshold: float = DEFAULT_DIST_THRESHOLD
    p_mismatch: float = DEFAULT_P_MISMATCH
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD


class PoCGenerateRequest(BaseModel):
    """Request for /generate endpoint."""
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    nonces: List[int]
    params: PoCParamsModel
    batch_size: int = 20
    wait: bool = False
    url: Optional[str] = None
    validation: Optional[ValidationModel] = None
    stat_test: Optional[StatTestModel] = None


# =============================================================================
# Helper Functions
# =============================================================================

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


def check_params_match(request: Request, params: PoCParamsModel):
    """Check if model matches deployed model. Raises 409 if mismatch."""
    # Get model names from openai_serving_models
    serving_models = getattr(request.app.state, 'openai_serving_models', None)
    if not serving_models or not hasattr(serving_models, 'base_model_paths'):
        return  # No model info available
    
    base_paths = serving_models.base_model_paths
    if not base_paths:
        return
    
    model_path = base_paths[0].model_path
    served_names = [p.name for p in base_paths]
    
    # Accept model path OR any served model name
    valid_models = {model_path} | set(served_names)
    if params.model not in valid_models:
        raise HTTPException(
            status_code=409,
            detail=f"model mismatch: requested={params.model}, valid={list(valid_models)}"
        )


def _is_generation_active(app_id: int) -> bool:
    """Check if a generation loop is currently active for the given app."""
    tasks = _poc_tasks.get(app_id)
    if not tasks:
        return False
    gen_task = tasks.get("gen_task")
    if gen_task is None:
        return False
    return not gen_task.done()


def _get_api_status(app_id: int) -> dict:
    """Get PoC status from API-owned state."""
    tasks = _poc_tasks.get(app_id)
    
    if not tasks or not _is_generation_active(app_id):
        return {
            "status": PoCState.IDLE.value,
            "config": None,
            "stats": None,
        }
    
    config = tasks.get("config", {})
    stats = tasks.get("stats", {})
    start_time = stats.get("start_time", 0)
    total_processed = stats.get("total_processed", 0)
    elapsed = time.time() - start_time if start_time > 0 else 0
    nonces_per_second = total_processed / elapsed if elapsed > 0 else 0
    
    return {
        "status": PoCState.GENERATING.value,
        "config": {
            "block_hash": config.get("block_hash"),
            "block_height": config.get("block_height"),
            "public_key": config.get("public_key"),
            "node_id": config.get("node_id"),
            "node_count": config.get("node_count"),
            "seq_len": config.get("seq_len"),
            "k_dim": config.get("k_dim"),
        },
        "stats": {
            "total_processed": total_processed,
            "nonces_per_second": nonces_per_second,
        },
    }


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


# =============================================================================
# Background Tasks for /init/generate
# =============================================================================

def _get_next_nonces(nonce_counter: int, batch_size: int, node_count: int) -> tuple:
    """Generate next batch of nonces (API-side).
    
    Returns:
        (nonces_list, new_nonce_counter)
    """
    nonces = []
    counter = nonce_counter
    for _ in range(batch_size):
        nonces.append(counter)
        counter += node_count
    return nonces, counter


async def _generation_loop(
    engine_client,
    stop_event: asyncio.Event,
    artifact_queue: asyncio.Queue,
    config: dict,
    stats: dict,
):
    """Continuous generation loop for /init/generate.
    
    Computes nonces in API layer and calls engine's generate_artifacts.
    """
    # Initialize nonce counter: start at node_id, stride by node_count
    nonce_counter = config["node_id"]
    batch_size = config["batch_size"]
    node_count = config["node_count"]
    
    start_time = time.time()
    stats["start_time"] = start_time
    stats["total_processed"] = 0
    last_report_time = start_time
    
    logger.info("PoC generation started")
    
    skip_count = 0
    timeout_count = 0
    
    try:
        while not stop_event.is_set():
            # Generate next batch of nonces (API-side)
            nonces, nonce_counter = _get_next_nonces(nonce_counter, batch_size, node_count)
            
            try:
                result = await engine_client.poc_request(
                    "generate_artifacts",
                    {
                        "nonces": nonces,
                        "block_hash": config["block_hash"],
                        "public_key": config["public_key"],
                        "seq_len": config["seq_len"],
                        "k_dim": config["k_dim"],
                    },
                    timeout_ms=POC_RPC_TIMEOUT_MS
                )
                timeout_count = 0  # Reset on successful RPC
            except TimeoutError:
                # Timeout is recoverable - engine is busy with chat inference
                timeout_count += 1
                if timeout_count == 1 or timeout_count % 10 == 0:
                    logger.warning(
                        f"PoC generate_artifacts timed out (#{timeout_count}), "
                        "engine busy with inference. Retrying..."
                    )
                # Roll back nonce counter since this batch wasn't processed
                nonce_counter -= batch_size * node_count
                await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC * 2)
                continue
            
            # Chat-priority: if skipped due to engine busy, backoff and retry
            if result.get("skipped"):
                skip_count += 1
                if skip_count % 100 == 1:  # Log every 100 skips (~5s at 50ms backoff)
                    logger.debug(f"PoC yielding to chat (skip #{skip_count})")
                # Roll back nonce counter since this batch wasn't processed
                nonce_counter -= batch_size * node_count
                await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC)
                continue
            
            skip_count = 0  # Reset on successful batch
            artifacts = result.get("artifacts", [])
            
            if artifacts:
                # Convert dict artifacts to Artifact objects for queue
                artifact_objs = [
                    Artifact(nonce=a["nonce"], vector_b64=a["vector_b64"])
                    for a in artifacts
                ]
                await artifact_queue.put({
                    "public_key": config["public_key"],
                    "block_hash": config["block_hash"],
                    "block_height": config["block_height"],
                    "node_id": config["node_id"],
                    "artifacts": artifact_objs,
                })
            
            stats["total_processed"] += len(nonces)
            
            # Log progress every 5 seconds
            current_time = time.time()
            if current_time - last_report_time >= 5.0:
                elapsed_min = (current_time - start_time) / 60
                rate = stats["total_processed"] / elapsed_min if elapsed_min > 0 else 0
                logger.info(f"Generated: {stats['total_processed']} nonces in {elapsed_min:.2f}min ({rate:.0f}/min)")
                last_report_time = current_time
            
    except asyncio.CancelledError:
        elapsed_min = (time.time() - start_time) / 60
        logger.info(f"PoC stopped: {stats['total_processed']} nonces in {elapsed_min:.2f}min")
    except Exception as e:
        elapsed_min = (time.time() - start_time) / 60
        logger.error(
            f"PoC generation loop crashed after {stats['total_processed']} nonces "
            f"in {elapsed_min:.2f}min: {e}",
            exc_info=True
        )
        raise


async def _callback_sender_loop(
    artifact_queue: asyncio.Queue,
    callback_url: str,
    stop_event: asyncio.Event,
    k_dim: int,
):
    """Batches artifacts and sends callbacks every POC_CALLBACK_INTERVAL_SEC."""
    accumulated: List[Artifact] = []
    last_send_time = time.time()
    metadata = {}
    
    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            try:
                # Wait for artifacts with timeout
                batch = await asyncio.wait_for(
                    artifact_queue.get(),
                    timeout=1.0
                )
                accumulated.extend(batch["artifacts"])
                metadata = {
                    "public_key": batch["public_key"],
                    "block_hash": batch["block_hash"],
                    "block_height": batch["block_height"],
                    "node_id": batch["node_id"],
                }
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break
            
            # Send if interval elapsed and we have data
            current_time = time.time()
            should_send = (
                accumulated and 
                (current_time - last_send_time >= POC_CALLBACK_INTERVAL_SEC or stop_event.is_set())
            )
            
            if should_send:
                payload = {
                    **metadata,
                    "artifacts": [{"nonce": a.nonce, "vector_b64": a.vector_b64} for a in accumulated],
                    "encoding": {"dtype": "f16", "k_dim": k_dim, "endian": "le"},
                }
                
                try:
                    await session.post(
                        f"{callback_url}/generated",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    )
                    logger.debug(f"Callback sent: {len(accumulated)} artifacts to {callback_url}/generated")
                except Exception as e:
                    logger.warning(f"Callback failed: {e}")
                
                accumulated = []
                last_send_time = current_time
    
    # Send remaining on shutdown
    if accumulated and callback_url:
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    **metadata,
                    "artifacts": [{"nonce": a.nonce, "vector_b64": a.vector_b64} for a in accumulated],
                    "encoding": {"dtype": "f16", "k_dim": k_dim, "endian": "le"},
                }
                await session.post(
                    f"{callback_url}/generated",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                )
        except Exception:
            pass


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/init/generate")
async def init_generate(request: Request, body: PoCInitGenerateRequest) -> dict:
    """Initialize PoC round and start continuous generation.
    
    Callbacks sent to {url}/generated every POC_CALLBACK_INTERVAL_SEC seconds.
    All state (nonce counter, stats) is managed in the API layer.
    """
    await check_poc_enabled(request)
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)
    
    app_id = id(request.app)
    
    # Check for conflicts (API-owned state)
    if _is_generation_active(app_id):
        raise HTTPException(status_code=409, detail="Already generating")
    
    # Cancel any lingering tasks
    await _cancel_poc_tasks(app_id)
    
    # Build config
    config = {
        "block_hash": body.block_hash,
        "block_height": body.block_height,
        "public_key": body.public_key,
        "node_id": body.node_id,
        "node_count": body.node_count,
        "batch_size": body.batch_size,
        "seq_len": body.params.seq_len,
        "k_dim": body.params.k_dim,
    }
    
    # Shared stats dict (updated by generation loop)
    stats = {"start_time": 0, "total_processed": 0}
    
    # Start background tasks
    stop_event = asyncio.Event()
    artifact_queue: asyncio.Queue = asyncio.Queue()
    
    gen_task = asyncio.create_task(
        _generation_loop(engine_client, stop_event, artifact_queue, config, stats)
    )
    
    send_task = None
    if body.url:
        send_task = asyncio.create_task(
            _callback_sender_loop(artifact_queue, body.url, stop_event, body.params.k_dim)
        )
    
    _poc_tasks[app_id] = {
        "gen_task": gen_task,
        "send_task": send_task,
        "stop_event": stop_event,
        "queue": artifact_queue,
        "config": config,
        "stats": stats,
    }
    
    return {
        "status": "OK",
        "pow_status": {"status": "GENERATING"},
    }


@router.post("/generate")
async def generate(request: Request, body: PoCGenerateRequest) -> dict:
    """Compute artifacts for specific nonces. Optionally validate against provided artifacts.
    
    - wait=true: process synchronously and return result
    - wait=false: queue job and return request_id (poll GET /generate/{request_id} for result)
    
    If /init/generate is running, job is queued and waits until it's idle.
    """
    await check_poc_enabled(request)
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)
    
    app_id = id(request.app)
    
    # Validate nonce set match if validation provided
    if body.validation:
        validation_nonces = set(a.nonce for a in body.validation.artifacts)
        request_nonces = set(body.nonces)
        if validation_nonces != request_nonces:
            raise HTTPException(
                status_code=400,
                detail="validation.artifacts nonces must match nonces field exactly"
            )
    
    # Build validation map if provided
    validation_map = None
    if body.validation:
        validation_map = {a.nonce: a.vector_b64 for a in body.validation.artifacts}
    
    # Get stat_test params
    stat_test = body.stat_test or StatTestModel()
    
    # wait=false: enqueue and return immediately
    if not body.wait:
        request_id = str(uuid.uuid4())
        
        job = GenerateJob(
            request_id=request_id,
            engine_client=engine_client,
            app_id=app_id,
            block_hash=body.block_hash,
            block_height=body.block_height,
            public_key=body.public_key,
            node_id=body.node_id,
            node_count=body.node_count,
            nonces=body.nonces,
            seq_len=body.params.seq_len,
            k_dim=body.params.k_dim,
            batch_size=body.batch_size,
            validation_artifacts=validation_map,
            stat_test_dist_threshold=stat_test.dist_threshold,
            stat_test_p_mismatch=stat_test.p_mismatch,
            stat_test_fraud_threshold=stat_test.fraud_threshold,
            callback_url=body.url,
        )
        
        # Store initial result record
        _generate_results[request_id] = GenerateResult(status="queued")
        
        # Ensure worker is running and enqueue
        await _ensure_worker_running(engine_client, app_id)
        await _generate_queue.put(job)
        
        return {
            "status": "queued",
            "request_id": request_id,
            "queued_count": len(body.nonces),
        }
    
    # wait=true: process synchronously (existing logic)
    # Wait if /init/generate is active
    while _is_generation_active(app_id):
        await asyncio.sleep(0.1)
    
    computed_artifacts = []
    batch_size = body.batch_size
    
    for i in range(0, len(body.nonces), batch_size):
        chunk = body.nonces[i:i + batch_size]
        chunk_start_time = time.time()
        
        while True:
            # Wait if /init/generate became active
            while _is_generation_active(app_id):
                await asyncio.sleep(0.1)
            
            result = await engine_client.poc_request("generate_artifacts", {
                "nonces": chunk,
                "block_hash": body.block_hash,
                "public_key": body.public_key,
                "seq_len": body.params.seq_len,
                "k_dim": body.params.k_dim,
            })
            
            if not result.get("skipped"):
                computed_artifacts.extend(result.get("artifacts", []))
                break
            
            elapsed = time.time() - chunk_start_time
            if elapsed >= POC_GENERATE_CHUNK_TIMEOUT_SEC:
                raise HTTPException(
                    status_code=503,
                    detail=f"Timeout waiting for engine: chunk {i//batch_size} "
                           f"timed out after {elapsed:.1f}s"
                )
            
            await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC)
    
    # If no validation, return computed artifacts
    if not body.validation:
        return {
            "status": "completed",
            "request_id": str(uuid.uuid4()),
            "artifacts": computed_artifacts,
            "encoding": {"dtype": "f16", "k_dim": body.params.k_dim, "endian": "le"},
        }
    
    # Validation mode: compare computed vs received
    n_mismatch = 0
    mismatch_nonces = []
    
    for artifact in computed_artifacts:
        nonce = artifact["nonce"]
        computed_b64 = artifact["vector_b64"]
        received_b64 = validation_map.get(nonce)
        
        if received_b64:
            computed_vec = decode_vector(computed_b64)
            received_vec = decode_vector(received_b64)
            distance = np.linalg.norm(computed_vec - received_vec)
            
            if distance > stat_test.dist_threshold:
                n_mismatch += 1
                mismatch_nonces.append(nonce)
    
    n_total = len(body.nonces)
    p_value, fraud_detected = fraud_test(
        n_mismatch, n_total,
        stat_test.p_mismatch, stat_test.fraud_threshold
    )
    
    response = {
        "status": "completed",
        "request_id": str(uuid.uuid4()),
        "n_total": n_total,
        "n_mismatch": n_mismatch,
        "mismatch_nonces": mismatch_nonces,
        "p_value": p_value,
        "fraud_detected": fraud_detected,
    }
    
    # Send callback if URL provided
    if body.url:
        try:
            async with aiohttp.ClientSession() as session:
                callback_payload = {
                    "request_id": response["request_id"],
                    "block_hash": body.block_hash,
                    "block_height": body.block_height,
                    "public_key": body.public_key,
                    "node_id": body.node_id,
                    "n_total": n_total,
                    "n_mismatch": n_mismatch,
                    "mismatch_nonces": mismatch_nonces,
                    "p_value": p_value,
                    "fraud_detected": fraud_detected,
                }
                await session.post(
                    f"{body.url}/validated",
                    json=callback_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                )
        except Exception as e:
            logger.warning(f"Validation callback failed: {e}")
    
    return response


@router.get("/generate/{request_id}")
async def get_generate_result(request: Request, request_id: str) -> dict:
    """Poll for result of a queued /generate request.
    
    Returns:
        - status: "queued" | "running" | "completed" | "failed"
        - For "completed": same payload as synchronous /generate
        - For "failed": error message
    """
    await check_poc_enabled(request)
    
    record = _generate_results.get(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    
    response = {"status": record.status, "request_id": request_id}
    
    if record.status == "completed" and record.result:
        response.update(record.result)
    elif record.status == "failed" and record.error:
        response["error"] = record.error
    
    return response


@router.get("/status")
async def get_status(request: Request) -> dict:
    """Get current PoC status (API-owned state)."""
    await check_poc_enabled(request)
    
    app_id = id(request.app)
    return _get_api_status(app_id)


@router.post("/stop")
async def stop_round(request: Request) -> dict:
    """Stop current PoC round (cancels API background tasks)."""
    await check_poc_enabled(request)
    
    app_id = id(request.app)
    
    # Cancel background tasks
    await _cancel_poc_tasks(app_id)
    
    return {
        "status": "OK", 
        "pow_status": {"status": "STOPPED"},
    }
