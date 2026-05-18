"""Run a minimal Phase 15 SAINT train-only step on a large HF model."""

from __future__ import annotations

import argparse
from json import dumps
from pathlib import Path
from time import perf_counter
from typing import Any

from saint.config import RuntimeConfig
from saint.runtime import train_runtime


def _texts(path: str | None, limit: int, *, offset: int = 0) -> list[str]:
    if not path:
        return ["simple ai node training"]
    source = Path(path)
    if not source.exists():
        return ["simple ai node training"]
    text = source.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        start = max(0, offset)
        selected = lines[start:start + max(1, limit)]
        return selected or lines[: max(1, limit)]
    return [text[:256] or "simple ai node training"]


def _target_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _hf_metadata(args) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "hf_device_map": args.hf_device_map,
            "hf_max_memory": args.hf_max_memory,
            "hf_offload_folder": args.hf_offload_folder,
        }.items()
        if value
    }


def _config(args) -> RuntimeConfig:
    metadata: dict[str, Any] = {
        "model_name_or_path": args.model,
        "device": args.device,
        "model_dtype": args.model_dtype,
        "texts": _texts(args.corpus, args.train_texts),
        "validation_texts": _texts(
            args.corpus,
            args.validation_texts,
            offset=args.train_texts,
        ),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lr_decay": args.lr_decay,
        "routing_method": args.routing_method,
        "routing_max_length": args.routing_max_length,
        "routing_batch_size": args.routing_batch_size,
        "delta_application": "inplace",
        "train_only": True,
        "gradient_checkpointing": args.gradient_checkpointing,
        "measure_train_only_loss": args.measure_loss,
        "validate_during_train": args.validate_during_train,
        "early_stopping": args.early_stopping,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "validation_rerank_multiplier": args.validation_rerank_multiplier,
        "validation_rerank_chunk_size": args.validation_rerank_chunk_size,
        "validation_probe_epsilon": args.validation_probe_epsilon,
        "target_names": _target_names(args.target_names),
        "target_device": args.target_device,
        "max_cuda_gb": args.max_cuda_gb,
        "marco": "fase_15_marco_3",
        **_hf_metadata(args),
    }
    return RuntimeConfig(
        experiment_name="phase15_14b_train_only",
        output_dir=args.out,
        task="huggingface_causal_lm",
        method="hf_saint_forward_smoke",
        seed=args.seed,
        steps=args.steps,
        parameter_budget=args.budget,
        vram_gb=args.max_cuda_gb,
        metadata=metadata,
    )


def run(args) -> dict[str, Any]:
    start = perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint = train_runtime(_config(args))
        result = {
            "status": "ok",
            "elapsed_s": perf_counter() - start,
            "checkpoint": checkpoint,
        }
    except Exception as exc:  # pragma: no cover - used for large-model diagnostics.
        result = {
            "status": "failed",
            "elapsed_s": perf_counter() - start,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    (out / "phase15_train_only_result.json").write_text(
        dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", default="data/tinyshakespeare_phase13.txt")
    parser.add_argument("--out", default="runs/phase15_train_only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-texts", type=int, default=1)
    parser.add_argument("--validation-texts", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--routing-method", default="activation")
    parser.add_argument("--routing-max-length", type=int, default=4)
    parser.add_argument("--routing-batch-size", type=int, default=1)
    parser.add_argument(
        "--target-names",
        default="model.layers.0.self_attn.q_proj.weight",
    )
    parser.add_argument("--target-device", default="cuda")
    parser.add_argument("--max-cuda-gb", type=float, default=23.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--measure-loss", action="store_true")
    parser.add_argument("--validate-during-train", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--validation-rerank-multiplier", type=int, default=4)
    parser.add_argument("--validation-rerank-chunk-size", type=int, default=256)
    parser.add_argument("--validation-probe-epsilon", type=float, default=1e-3)
    parser.add_argument("--hf-device-map", default=None)
    parser.add_argument("--hf-max-memory", default=None)
    parser.add_argument("--hf-offload-folder", default=None)
    args = parser.parse_args()
    print(dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
