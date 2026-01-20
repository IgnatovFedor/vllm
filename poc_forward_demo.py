"""
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export GPU_NAME=4070s
python poc_forward_demo.py --mode generate --model Qwen/Qwen3-0.6B-FP8 --batch-size 8 -o ${GPU_NAME}_fp8
python poc_forward_demo.py --mode generate --model RedHatAI/Qwen3-0.6B-quantized.w4a16 --batch-size 8 -o ${GPU_NAME}_int4
python poc_forward_demo.py --mode compare --file-a ${GPU_NAME}_fp8.npz --file-b ${GPU_NAME}_int4.npz
"""

import argparse

import numpy as np

from vllm.engine.arg_utils import EngineArgs
from vllm.poc.poc_model_runner import execute_poc_forward
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor import Executor


def generate_mode(model_name: str, batch_size: int, output: str) -> None:
    max_model_len = 256
    gpu_memory_utilization = 0.6

    block_hash = "block_alpha"
    public_key = "node_A"

    total_nonces = 3000
    seq_len = 256
    k_dim = 12

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)
    executor_class = Executor.get_class(vllm_config)

    engine = LLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=not engine_args.disable_log_stats,
        usage_context=UsageContext.ENGINE_CONTEXT,
        multiprocess_mode=False,
    )

    model_executor = engine.model_executor
    model_config = engine.model_config
    hidden_size = model_config.get_hidden_size()

    all_nonces = list(range(total_nonces))
    all_vectors = np.zeros((total_nonces, k_dim), dtype=np.float16)
    all_last_hidden = np.zeros((total_nonces, hidden_size), dtype=np.float16)

    idx = 0
    while idx < total_nonces:
        batch_nonces = all_nonces[idx : idx + batch_size]

        results = model_executor.collective_rpc(
            execute_poc_forward,
            args=(
                block_hash,
                public_key,
                batch_nonces,
                seq_len,
                hidden_size,
                k_dim,
            ),
        )
        result = next((r for r in results if r is not None), None)

        if result is None:
            raise RuntimeError("execute_poc_forward returned None for batch")

        batch_vectors = result["vectors"]       # [len(batch_nonces), k_dim]
        batch_last_hidden = result["last_hidden"]  # [len(batch_nonces), hidden_size]

        # Map each nonce to its slot in the global arrays
        for i, nonce in enumerate(result["nonces"]):
            all_vectors[nonce, :] = batch_vectors[i]
            all_last_hidden[nonce, :] = batch_last_hidden[i]

        idx += batch_size

    np.savez(
        f"{output}.npz",
        nonces=np.array(all_nonces, dtype=np.int64),
        vectors=all_vectors,
        last_hidden=all_last_hidden,
    )
    print(f"Saved vectors to {output}.npz")


def _load_npz(path: str):
    data = np.load(path)
    nonces = data["nonces"]
    vectors = data["vectors"]
    last_hidden = data["last_hidden"]
    return nonces, vectors, last_hidden


def compare_mode(file_a: str, file_b: str) -> None:
    nonces_a, vec_a, hid_a = _load_npz(file_a)
    nonces_b, vec_b, hid_b = _load_npz(file_b)

    # Align by nonce
    idx_a = {int(n): i for i, n in enumerate(nonces_a)}
    idx_b = {int(n): i for i, n in enumerate(nonces_b)}
    common = sorted(set(idx_a.keys()) & set(idx_b.keys()))

    if not common:
        raise RuntimeError("No overlapping nonces between the two files")

    va = np.stack([vec_a[idx_a[n]] for n in common], axis=0)
    vb = np.stack([vec_b[idx_b[n]] for n in common], axis=0)
    ha = np.stack([hid_a[idx_a[n]] for n in common], axis=0)
    hb = np.stack([hid_b[idx_b[n]] for n in common], axis=0)

    vec_l2 = np.linalg.norm(va - vb, axis=1)
    hid_l2 = np.linalg.norm(ha - hb, axis=1)

    print(f"Common nonces: {len(common)}")
    print("Vectors L2:  mean={:.6f}, std={:.6f}".format(vec_l2.mean(), vec_l2.std()))
    print("Hidden  L2:  mean={:.6f}, std={:.6f}".format(hid_l2.mean(), hid_l2.std()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["generate", "compare"],
        default="generate",
        help="generate: run model and save npz; compare: compare two npz files",
    )
    parser.add_argument("--file-a", type=str, help="First npz file for compare mode")
    parser.add_argument("--file-b", type=str, help="Second npz file for compare mode")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--output", "-o", type=str, help="Output file name for generated npz")
    args = parser.parse_args()

    if args.mode == "generate":
        generate_mode(args.model, args.batch_size, args.output)
    else:
        if not args.file_a or not args.file_b:
            raise SystemExit("compare mode requires --file-a and --file-b")
        compare_mode(args.file_a, args.file_b)


if __name__ == "__main__":
    main()
