"""Compare Phase 15 train-only SAINT budgets and a small LoRA baseline."""

from __future__ import annotations

import argparse
from json import dumps
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any


def _items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _memory_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def _ints(value: str) -> list[int]:
    return [int(item) for item in _items(value)]


def _gb(value: int) -> float:
    return value / 1_000_000_000


def _row_from_saint(result: dict[str, Any], *, budget: int, max_memory: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": "saint_train_only",
        "budget": budget,
        "max_memory": max_memory,
        "status": result["status"],
        "elapsed_s": result["elapsed_s"],
    }
    if result["status"] != "ok":
        row.update({"error": result.get("error"), "train_cuda_gb": None})
        return row
    checkpoint = result["checkpoint"]
    metadata = checkpoint["metadata"]
    row.update(
        {
            "train_loss": checkpoint["train_loss"],
            "parameter_count": checkpoint["parameter_count"],
            "gain_per_parameter": 0.0,
            "tokens_per_s": metadata["tokens_per_s"],
            "load_s": metadata["stage_elapsed"]["load_s"],
            "routing_s": metadata["stage_elapsed"]["routing_s"],
            "train_s": metadata["stage_elapsed"]["train_s"],
            "checkpoint_payload_s": metadata["stage_elapsed"]["checkpoint_payload_s"],
            "load_cuda_gb": _gb(metadata["load_cuda_peak_bytes"]),
            "routing_cuda_gb": _gb(metadata["routing_cuda_peak_bytes"]),
            "train_cuda_gb": _gb(metadata["train_cuda_peak_bytes"]),
            "checkpoint_bytes": sum(item["bytes"] for item in checkpoint["files"]),
        }
    )
    return row


def _saint_args(args, *, budget: int, max_memory: str) -> SimpleNamespace:
    label = max_memory.replace("=", "_").replace(",", "_").replace(":", "_")
    return SimpleNamespace(
        model=args.model,
        corpus=args.corpus,
        out=str(Path(args.out) / f"saint_b{budget}_{label}"),
        device=args.device,
        model_dtype=args.model_dtype,
        seed=args.seed,
        steps=args.steps,
        budget=budget,
        batch_size=args.batch_size,
        train_texts=args.train_texts,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        routing_method=args.routing_method,
        routing_max_length=args.routing_max_length,
        routing_batch_size=args.routing_batch_size,
        target_names=args.target_names,
        target_device=args.target_device,
        max_cuda_gb=args.max_cuda_gb,
        gradient_checkpointing=args.gradient_checkpointing,
        hf_device_map=args.hf_device_map,
        hf_max_memory=max_memory,
        hf_offload_folder=str(Path(args.out) / f"offload_{label}"),
    )


def _run_saint_subprocess(args, *, budget: int, max_memory: str) -> dict[str, Any]:
    values = _saint_args(args, budget=budget, max_memory=max_memory)
    script = Path(__file__).with_name("benchmark_huggingface_phase15_train_only.py")
    command = [
        sys.executable,
        str(script),
        "--model",
        values.model,
        "--corpus",
        values.corpus,
        "--out",
        values.out,
        "--device",
        values.device,
        "--model-dtype",
        values.model_dtype,
        "--seed",
        str(values.seed),
        "--steps",
        str(values.steps),
        "--budget",
        str(values.budget),
        "--batch-size",
        str(values.batch_size),
        "--train-texts",
        str(values.train_texts),
        "--max-length",
        str(values.max_length),
        "--learning-rate",
        str(values.learning_rate),
        "--routing-method",
        values.routing_method,
        "--routing-max-length",
        str(values.routing_max_length),
        "--routing-batch-size",
        str(values.routing_batch_size),
        "--target-names",
        values.target_names,
        "--target-device",
        values.target_device,
        "--max-cuda-gb",
        str(values.max_cuda_gb),
        "--hf-device-map",
        values.hf_device_map,
        "--hf-max-memory",
        values.hf_max_memory,
        "--hf-offload-folder",
        values.hf_offload_folder,
    ]
    if values.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    result_path = Path(values.out) / "phase15_train_only_result.json"
    if result_path.exists():
        from json import loads

        return loads(result_path.read_text(encoding="utf-8"))
    return {
        "status": "failed",
        "elapsed_s": 0.0,
        "error": completed.stderr[-1000:] or completed.stdout[-1000:],
    }


def _load_batch(tokenizer, text: str, *, max_length: int, device):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    encoded = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    mask = encoded.get("attention_mask")
    return encoded["input_ids"].to(device), mask.to(device) if mask is not None else None


def _plain_loss(model, input_ids, attention_mask):
    kwargs = {"input_ids": input_ids, "labels": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return model(**kwargs).loss


def _lora_rank1(args) -> dict[str, Any]:
    from saint.adapters.huggingface_loading import load_causal_lm, model_dtype
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    metadata = {
        "model_dtype": args.model_dtype,
        "hf_device_map": args.hf_device_map,
        "hf_max_memory": args.lora_max_memory,
        "hf_offload_folder": str(Path(args.out) / "offload_lora_rank1"),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = load_causal_lm(AutoModelForCausalLM, args.model, device, metadata)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_ids, attention_mask = _load_batch(
        tokenizer,
        "simple ai node training",
        max_length=args.max_length,
        device=device,
    )
    named = dict(model.named_parameters())
    target = _items(args.target_names)[0]
    if target not in named:
        raise ValueError(f"missing LoRA target: {target}")
    for param in model.parameters():
        param.requires_grad_(False)
    weight = named[target]
    rows, cols = weight.shape
    dtype = model_dtype(torch, args.model_dtype) or weight.dtype
    a = (torch.randn(rows, 1, device=weight.device, dtype=dtype) * 0.01).requires_grad_()
    b = torch.zeros(1, cols, device=weight.device, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.AdamW([a, b], lr=args.lora_learning_rate)
    model.train()
    from time import perf_counter

    start = perf_counter()
    update = None
    for _ in range(args.steps):
        optimizer.zero_grad()
        update = a @ b
        with torch.no_grad():
            weight.add_(update)
        try:
            loss = _plain_loss(model, input_ids, attention_mask)
            loss.backward()
        finally:
            with torch.no_grad():
                weight.sub_(update)
        optimizer.step()
    elapsed = perf_counter() - start
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    if peak / 1_000_000_000 > args.max_cuda_gb:
        raise RuntimeError(f"CUDA budget exceeded during lora: {peak / 1_000_000_000:.3f} GB")
    return {
        "method": "lora_rank1_train_only",
        "budget": None,
        "rank": 1,
        "max_memory": args.lora_max_memory,
        "status": "ok",
        "elapsed_s": elapsed,
        "train_loss": float(loss.detach().cpu().item()),
        "parameter_count": int(a.numel() + b.numel()),
        "gain_per_parameter": 0.0,
        "train_cuda_gb": _gb(peak),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| method | budget | max_memory | status | train_s | train CUDA GB | params |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {budget} | {memory} | {status} | {train_s} | {cuda} | {params} |".format(
                method=row["method"],
                budget="" if row.get("budget") is None else row["budget"],
                memory=row.get("max_memory", ""),
                status=row["status"],
                train_s="" if row.get("train_s") is None else f"{row['train_s']:.3f}",
                cuda="" if row.get("train_cuda_gb") is None else f"{row['train_cuda_gb']:.3f}",
                params="" if row.get("parameter_count") is None else row["parameter_count"],
            )
        )
    return "\n".join(lines) + "\n"


def run(args) -> dict[str, Any]:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for max_memory in _memory_items(args.max_memories):
        for budget in _ints(args.budgets):
            result = _run_saint_subprocess(
                args,
                budget=budget,
                max_memory=max_memory,
            )
            rows.append(_row_from_saint(result, budget=budget, max_memory=max_memory))
    if any(row["status"] == "ok" and (row.get("train_cuda_gb") or 99) <= args.max_cuda_gb for row in rows):
        try:
            rows.append(_lora_rank1(args))
        except Exception as exc:  # pragma: no cover - large-model diagnostic.
            rows.append(
                {
                    "method": "lora_rank1_train_only",
                    "budget": None,
                    "max_memory": args.lora_max_memory,
                    "status": "failed",
                    "error": str(exc),
                    "train_cuda_gb": None,
                }
            )
    result = {"model": args.model, "rows": rows}
    (root / "phase15_compare_results.json").write_text(
        dumps(result, indent=2),
        encoding="utf-8",
    )
    (root / "phase15_compare_results.md").write_text(_markdown(rows), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", default="data/tinyshakespeare_phase13.txt")
    parser.add_argument("--out", default="runs/phase15_marco4_compare")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--budgets", default="1024,4096,8192")
    parser.add_argument(
        "--max-memories",
        default="0=12GiB,cpu=64GiB;0=14GiB,cpu=64GiB;0=16GiB,cpu=64GiB",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-texts", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--routing-method", default="activation")
    parser.add_argument("--routing-max-length", type=int, default=4)
    parser.add_argument("--routing-batch-size", type=int, default=1)
    parser.add_argument("--target-names", default="model.layers.0.self_attn.q_proj.weight")
    parser.add_argument("--target-device", default="cuda")
    parser.add_argument("--max-cuda-gb", type=float, default=23.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--hf-device-map", default="auto")
    parser.add_argument("--lora-max-memory", default="0=14GiB,cpu=64GiB")
    parser.add_argument("--lora-learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    print(dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
