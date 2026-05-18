"""Baseline comparison utilities for small Hugging Face SAINT runs."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from saint.config import RuntimeConfig


def _require_deps():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and transformers are required for Hugging Face benchmarks."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _texts() -> list[str]:
    return [
        "simple ai node training",
        "saint trains compact deltas",
        "small local causal language model",
    ]


def _batch(tokenizer, device, *, max_length: int):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    encoded = tokenizer(
        _texts(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask")
    return input_ids, mask.to(device) if mask is not None else None


def _loss(model, input_ids, attention_mask):
    kwargs = {"input_ids": input_ids, "labels": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return model(**kwargs).loss


def _full_finetune(
    model_path: str | Path,
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    device_name: str,
    max_length: int,
) -> dict[str, Any]:
    torch, AutoModelForCausalLM, AutoTokenizer = _require_deps()
    torch.manual_seed(seed)
    device = _device(torch, device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    input_ids, attention_mask = _batch(tokenizer, device, max_length=max_length)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    initial_loss = float(_loss(model, input_ids, attention_mask).detach().cpu().item())
    train_start = perf_counter()
    for _ in range(max(1, steps)):
        optimizer.zero_grad()
        loss = _loss(model, input_ids, attention_mask)
        loss.backward()
        optimizer.step()
    elapsed = max(perf_counter() - train_start, 1e-9)
    final_loss = float(_loss(model, input_ids, attention_mask).detach().cpu().item())
    tokens_seen = int(input_ids.numel()) * max(1, steps)
    cuda_peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    return {
        "method": "hf_full_finetune",
        "seed": seed,
        "initial_loss": initial_loss,
        "train_loss": final_loss,
        "loss_delta": final_loss - initial_loss,
        "parameter_count": sum(param.numel() for param in model.parameters()),
        "tokens_per_s": tokens_seen / elapsed,
        "tokens_seen": tokens_seen,
        "cuda_peak_bytes": cuda_peak,
        "device": str(device),
    }


def benchmark_hf_saint_vs_full(
    model_path: str | Path,
    run_dir: str | Path,
    *,
    seeds: tuple[int, ...] = (31, 32),
    steps: int = 2,
    parameter_budget: int = 8,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    max_length: int = 12,
) -> dict[str, Any]:
    from saint.runtime import merge_runtime, train_runtime

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        saint_dir = root / f"saint_seed_{seed}"
        config = RuntimeConfig(
            experiment_name=f"hf_saint_seed_{seed}",
            output_dir=str(saint_dir),
            task="huggingface_causal_lm",
            method="hf_saint_forward_smoke",
            steps=steps,
            parameter_budget=parameter_budget,
            seed=seed,
            metadata={
                "model_name_or_path": str(model_path),
                "checkpoint_dtype": "float16",
                "checkpoint_shard_bytes": 256,
                "device": device,
                "learning_rate": learning_rate,
                "max_length": max_length,
            },
        )
        saint = train_runtime(config)
        merged = merge_runtime(saint_dir)
        rows.append(
            {
                "method": "hf_saint_forward_smoke",
                "seed": seed,
                "initial_loss": saint["metadata"]["initial_loss"],
                "train_loss": saint["train_loss"],
                "loss_delta": saint["train_loss"] - saint["metadata"]["initial_loss"],
                "parameter_count": saint["parameter_count"],
                "tokens_per_s": saint["metadata"]["tokens_per_s"],
                "tokens_seen": saint["metadata"]["tokens_seen"],
                "cuda_peak_bytes": saint["metadata"]["cuda_peak_bytes"],
                "checkpoint_merge": bool(merged["merged"] and merged["shape_validation"]),
                "device": saint["metadata"]["device"],
            }
        )
        rows.append(
            _full_finetune(
                model_path,
                seed=seed,
                steps=steps,
                learning_rate=learning_rate,
                device_name=device,
                max_length=max_length,
            )
        )
    return {
        "model_path": str(model_path),
        "seeds": list(seeds),
        "steps": steps,
        "parameter_budget": parameter_budget,
        "rows": rows,
    }


__all__ = ["benchmark_hf_saint_vs_full"]
