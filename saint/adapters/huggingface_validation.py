"""Validation-oriented Hugging Face benchmarks for Phase 13 Marco 7."""

from __future__ import annotations

from json import dumps
from math import exp
from pathlib import Path
from typing import Any

from saint.adapters.huggingface_benchmark import (
    _batch,
    _full_finetune,
    _gain_per_parameter,
    _lora_finetune,
    _loss,
    _require_deps,
)
from saint.config import RuntimeConfig


def load_text_corpus(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    texts = [line.strip() for line in lines if line.strip()]
    if len(texts) < 2:
        raise ValueError("corpus must contain at least two non-empty lines")
    return texts


def split_texts(texts: list[str], *, validation_ratio: float = 0.25):
    cutoff = max(1, int(round(len(texts) * (1.0 - validation_ratio))))
    cutoff = min(cutoff, len(texts) - 1)
    return texts[:cutoff], texts[cutoff:]


def _artifact_bytes(run_dir: Path) -> int:
    return sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())


def _cuda_peak_for(device_name: str, operation):
    torch, _, _, _ = _require_deps()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type != "cuda":
        return operation(), 0
    torch.cuda.reset_peak_memory_stats(device)
    value = operation()
    return value, int(torch.cuda.max_memory_allocated(device))


def _generate(
    model_path: str | Path,
    *,
    prompt: str,
    device_name: str,
    merged_weights: dict[str, list[list[float]]] | None = None,
) -> str:
    torch, _, AutoModelForCausalLM, AutoTokenizer = _require_deps()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if merged_weights is not None:
        state = model.state_dict()
        with torch.no_grad():
            for name, matrix in merged_weights.items():
                if name not in state or not matrix:
                    continue
                tensor = state[name]
                values = torch.tensor(matrix, dtype=tensor.dtype, device=device)
                rows = min(tensor.shape[0], values.shape[0])
                cols = min(tensor.shape[1], values.shape[1])
                tensor[:rows, :cols].copy_(values[:rows, :cols])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return str(tokenizer.decode(output[0], skip_special_tokens=True))


def _evaluate_merged(
    model_path: str | Path,
    merged_weights: dict[str, list[list[float]]],
    *,
    validation_texts: list[str],
    device_name: str,
    max_length: int,
) -> dict[str, float]:
    torch, _, AutoModelForCausalLM, AutoTokenizer = _require_deps()
    device = torch.device(device_name)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    state = model.state_dict()
    with torch.no_grad():
        for name, matrix in merged_weights.items():
            if name not in state or not matrix:
                continue
            tensor = state[name]
            values = torch.tensor(matrix, dtype=tensor.dtype, device=device)
            rows = min(tensor.shape[0], values.shape[0])
            cols = min(tensor.shape[1], values.shape[1])
            tensor[:rows, :cols].copy_(values[:rows, :cols])
    input_ids, attention_mask = _batch(
        tokenizer,
        device,
        max_length=max_length,
        texts=validation_texts,
    )
    loss = float(_loss(model, input_ids, attention_mask).detach().cpu().item())
    return {"merged_validation_loss": loss, "merged_perplexity": exp(min(loss, 20.0))}


def _saint_validation_row(
    model_path: str | Path,
    root: Path,
    *,
    seed: int,
    steps: int,
    budget: int,
    learning_rate: float,
    device: str,
    max_length: int,
    train_texts: list[str],
    validation_texts: list[str],
    batch_size: int,
    max_trainable_matrices: int = 2,
    routing_method: str = "gradient",
) -> dict[str, Any]:
    from saint.runtime import merge_runtime, resume_runtime, train_runtime

    run_dir = root / f"saint_budget_{budget}_seed_{seed}"
    config = RuntimeConfig(
        experiment_name=f"hf_validation_saint_{budget}_{seed}",
        output_dir=str(run_dir),
        task="huggingface_causal_lm",
        method="hf_saint_forward_smoke",
        steps=steps,
        parameter_budget=budget,
        seed=seed,
        metadata={
            "model_name_or_path": str(model_path),
            "checkpoint_dtype": "float16",
            "checkpoint_shard_bytes": 512,
            "device": device,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "texts": train_texts,
            "validation_texts": validation_texts,
            "batch_size": batch_size,
            "max_trainable_matrices": max_trainable_matrices,
            "routing_method": routing_method,
        },
    )
    result = train_runtime(config)
    resumed = resume_runtime(run_dir)
    target_matrices = set(result["metadata"].get("target_matrices", [])) or None
    merged, merge_cuda_peak = _cuda_peak_for(
        result["metadata"]["device"],
        lambda: merge_runtime(
            run_dir,
            matrix_names=target_matrices,
            write_artifact=False,
        ),
    )
    merged_eval = _evaluate_merged(
        model_path,
        merged["merged_weights"],
        validation_texts=validation_texts,
        device_name=result["metadata"]["device"],
        max_length=max_length,
    )
    initial = result["metadata"]["initial_loss"]
    final = result["train_loss"]
    artifact_bytes = _artifact_bytes(run_dir)
    stage_memory = {
        "load_cuda_peak_bytes": result["metadata"].get("load_cuda_peak_bytes", 0),
        "routing_cuda_peak_bytes": result["metadata"].get("routing_cuda_peak_bytes", 0),
        "train_cuda_peak_bytes": result["metadata"].get("train_cuda_peak_bytes", 0),
        "checkpoint_file_bytes": artifact_bytes,
        "merge_cuda_peak_bytes": merge_cuda_peak,
    }
    return {
        "method": "saint",
        "seed": seed,
        "budget": budget,
        "rank": None,
        "train_loss": final,
        "validation_loss": result["metadata"]["validation_loss"],
        **merged_eval,
        "parameter_count": result["parameter_count"],
        "gain_per_parameter": _gain_per_parameter(
            initial,
            final,
            result["parameter_count"],
        ),
        "artifact_bytes": artifact_bytes,
        "tokens_per_s": result["metadata"]["tokens_per_s"],
        "cuda_peak_bytes": result["metadata"]["cuda_peak_bytes"],
        "load_cuda_peak_bytes": stage_memory["load_cuda_peak_bytes"],
        "routing_cuda_peak_bytes": stage_memory["routing_cuda_peak_bytes"],
        "train_cuda_peak_bytes": stage_memory["train_cuda_peak_bytes"],
        "checkpoint_file_bytes": stage_memory["checkpoint_file_bytes"],
        "merge_cuda_peak_bytes": stage_memory["merge_cuda_peak_bytes"],
        "stage_memory": stage_memory,
        "delta_payload_format": result["metadata"].get("delta_payload_format"),
        "resume_quality_delta": abs(resumed["train_loss"] - final),
        "device": result["metadata"]["device"],
        "merged_weights": merged["merged_weights"],
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| method | budget | rank | params | val loss | merge ppl | artifact bytes | gain/param |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {budget} | {rank} | {params} | {val:.6f} | "
            "{ppl:.6f} | {bytes} | {gain:.8f} |".format(
                method=row["method"],
                budget="" if row.get("budget") is None else row["budget"],
                rank="" if row.get("rank") is None else row["rank"],
                params=row["parameter_count"],
                val=row["validation_loss"],
                ppl=row.get("merged_perplexity", exp(min(row["validation_loss"], 20.0))),
                bytes=row.get("artifact_bytes", 0),
                gain=row["gain_per_parameter"],
            )
        )
    return "\n".join(lines) + "\n"


def run_hf_phase13_validation(
    model_path: str | Path,
    corpus_path: str | Path,
    run_dir: str | Path,
    *,
    seed: int = 31,
    steps: int = 8,
    budget: int = 8,
    lora_rank: int = 4,
    saint_learning_rate: float = 1e-3,
    lora_learning_rate: float = 5e-3,
    full_learning_rate: float = 1e-4,
    device: str = "auto",
    max_length: int = 24,
    batch_size: int = 3,
) -> dict[str, Any]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_texts, validation_texts = split_texts(load_text_corpus(corpus_path))
    rows: list[dict[str, Any]] = []
    rows.append(
        _saint_validation_row(
            model_path,
            root,
            seed=seed,
            steps=steps,
            budget=budget,
            learning_rate=saint_learning_rate,
            device=device,
            max_length=max_length,
            train_texts=train_texts,
            validation_texts=validation_texts,
            batch_size=batch_size,
        )
    )
    lora_path = root / f"lora_rank_{lora_rank}.pt"
    lora = _lora_finetune(
        model_path,
        seed=seed,
        steps=steps,
        learning_rate=lora_learning_rate,
        device_name=device,
        max_length=max_length,
        rank=lora_rank,
        alpha=float(lora_rank),
        max_targets=2,
        texts=train_texts,
        validation_texts=validation_texts,
        artifact_path=lora_path,
        batch_size=batch_size,
    )
    lora.update({"method": "lora", "budget": None, "rank": lora_rank})
    rows.append(lora)
    full = _full_finetune(
        model_path,
        seed=seed,
        steps=steps,
        learning_rate=full_learning_rate,
        device_name=device,
        max_length=max_length,
        texts=train_texts,
        validation_texts=validation_texts,
        batch_size=batch_size,
    )
    full.update({"method": "full", "budget": None, "rank": None, "artifact_bytes": 0})
    rows.append(full)
    saint_weights = rows[0].pop("merged_weights")
    generation = {
        "prompt": "SAINT",
        "base": _generate(model_path, prompt="SAINT", device_name=rows[0].get("device", device)),
        "saint_merged": _generate(
            model_path,
            prompt="SAINT",
            device_name=rows[0].get("device", device),
            merged_weights=saint_weights,
        ),
    }
    result = {
        "model_path": str(model_path),
        "corpus_path": str(corpus_path),
        "train_examples": len(train_texts),
        "validation_examples": len(validation_texts),
        "steps": steps,
        "batch_size": batch_size,
        "rows": rows,
        "generation": generation,
    }
    (root / "validation_results.json").write_text(
        dumps(result, indent=2),
        encoding="utf-8",
    )
    (root / "validation_results.md").write_text(_markdown(rows), encoding="utf-8")
    return result


__all__ = ["load_text_corpus", "run_hf_phase13_validation", "split_texts"]
