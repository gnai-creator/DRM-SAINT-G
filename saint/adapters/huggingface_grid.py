"""Hyperparameter grid for Phase 13 Marco 8."""

from __future__ import annotations

from json import dumps
from math import exp
from pathlib import Path
from typing import Any

from saint.adapters.huggingface_benchmark import (
    _batch,
    _gain_per_parameter,
    _lora_finetune,
    _loss,
    _model_dtype,
    _require_deps,
)
from saint.adapters.huggingface_loading import load_causal_lm
from saint.adapters.huggingface_validation import (
    _generate,
    _saint_validation_row,
    load_text_corpus,
    split_texts,
)
from saint.checkpoints import require_sparse_delta_payload, validate_checkpoint_bundle


def _label(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _base_validation(
    model_path: str | Path,
    validation_texts: list[str],
    *,
    device_name: str,
    max_length: int,
    model_dtype: str | None = None,
    hf_load_metadata: dict[str, Any] | None = None,
) -> dict[str, float]:
    torch, _, AutoModelForCausalLM, AutoTokenizer = _require_deps()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    metadata = {"model_dtype": model_dtype, **(hf_load_metadata or {})}
    model = load_causal_lm(AutoModelForCausalLM, model_path, device, metadata)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    input_ids, attention_mask = _batch(
        tokenizer,
        device,
        max_length=max_length,
        texts=validation_texts,
    )
    loss = float(_loss(model, input_ids, attention_mask).detach().cpu().item())
    return {"base_validation_loss": loss, "base_perplexity": exp(min(loss, 20.0))}


def _write_saint_delta_only(run_dir: Path, out_path: Path) -> dict[str, Any]:
    checkpoint = validate_checkpoint_bundle(run_dir)
    payload = require_sparse_delta_payload(checkpoint, run_dir)
    sparse: dict[str, list[list[float]]] = {}
    value_count = 0
    for name, entries in payload.get("values", {}).items():
        kept = []
        for row_index, col_index, value in entries:
            if abs(float(value)) > 0.0:
                kept.append([int(row_index), int(col_index), float(value)])
        if kept:
            sparse[name] = kept
            value_count += len(kept)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dumps({"format": "saint_sparse_delta", "values": sparse}), encoding="utf-8")
    return {"delta_only_bytes": out_path.stat().st_size, "delta_only_values": value_count}


def _best(rows: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["method"] == method]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["validation_loss"])


def _markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "| method | budget | rank | lr | params | val loss | base delta | gain/param | bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {budget} | {rank} | {lr} | {params} | {val:.6f} | "
            "{delta:.6f} | {gain:.8f} | {bytes} |".format(
                method=row["method"],
                budget="" if row.get("budget") is None else row["budget"],
                rank="" if row.get("rank") is None else row["rank"],
                lr=row["learning_rate"],
                params=row["parameter_count"],
                val=row["validation_loss"],
                delta=row["validation_delta_vs_base"],
                gain=row["gain_per_parameter"],
                bytes=row.get("delta_only_bytes", row.get("artifact_bytes", 0)),
            )
        )
    lines.append("")
    lines.append("```json")
    lines.append(dumps(summary, indent=2))
    lines.append("```")
    return "\n".join(lines) + "\n"


def run_hf_phase13_grid(
    model_path: str | Path,
    corpus_path: str | Path,
    run_dir: str | Path,
    *,
    seed: int = 31,
    steps: int = 6,
    saint_budgets: tuple[int, ...] = (8, 16),
    saint_lrs: tuple[float, ...] = (1e-3, 5e-3),
    lora_ranks: tuple[int, ...] = (2, 4),
    lora_lrs: tuple[float, ...] = (1e-3, 5e-3),
    device: str = "auto",
    max_length: int = 24,
    batch_size: int = 3,
    saint_target_matrices: int = 2,
    saint_routing_method: str = "gradient",
    saint_routing_max_length: int | None = None,
    saint_routing_batch_size: int | None = None,
    model_dtype: str | None = None,
    max_cuda_gb: float | None = None,
    saint_delta_application: str = "functional",
    hf_load_metadata: dict[str, Any] | None = None,
    saint_target_names: tuple[str, ...] = (),
    saint_target_device: str | None = None,
    prompts: tuple[str, ...] = ("SAINT", "Checkpoint", "LoRA"),
) -> dict[str, Any]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_texts, validation_texts = split_texts(load_text_corpus(corpus_path))
    base = _base_validation(
        model_path,
        validation_texts,
        device_name=device,
        max_length=max_length,
        model_dtype=model_dtype,
        hf_load_metadata=hf_load_metadata,
    )
    rows: list[dict[str, Any]] = []
    generation: dict[str, dict[str, str]] = {}
    for budget in saint_budgets:
        for lr in saint_lrs:
            combo = root / f"saint_b{budget}_lr{_label(lr)}"
            row = _saint_validation_row(
                model_path,
                combo,
                seed=seed,
                steps=steps,
                budget=budget,
                learning_rate=lr,
                device=device,
                max_length=max_length,
                train_texts=train_texts,
                validation_texts=validation_texts,
                batch_size=batch_size,
                max_trainable_matrices=saint_target_matrices,
                routing_method=saint_routing_method,
                routing_max_length=saint_routing_max_length,
                routing_batch_size=saint_routing_batch_size,
                model_dtype=model_dtype,
                max_cuda_gb=max_cuda_gb,
                delta_application=saint_delta_application,
                hf_load_metadata=hf_load_metadata,
                target_names=saint_target_names,
                target_device=saint_target_device,
            )
            delta_payload = row.pop("merged_delta_payload")
            run_path = combo / f"saint_budget_{budget}_seed_{seed}"
            row.update(_write_saint_delta_only(run_path, combo / "saint_delta_only.json"))
            row["learning_rate"] = lr
            row["validation_delta_vs_base"] = row["validation_loss"] - base["base_validation_loss"]
            rows.append(row)
            if not generation:
                device_name = row.get("device", device)
                for prompt in prompts:
                    generation[prompt] = {
                        "base": _generate(
                            model_path,
                            prompt=prompt,
                            device_name=device_name,
                            model_dtype=model_dtype,
                        ),
                        "saint_merged": _generate(
                            model_path,
                            prompt=prompt,
                            device_name=device_name,
                            merged_delta_payload=delta_payload,
                            model_dtype=model_dtype,
                        ),
                    }
    for rank in lora_ranks:
        for lr in lora_lrs:
            artifact = root / f"lora_r{rank}_lr{_label(lr)}.pt"
            row = _lora_finetune(
                model_path,
                seed=seed,
                steps=steps,
                learning_rate=lr,
                device_name=device,
                max_length=max_length,
                rank=rank,
                alpha=float(rank),
                max_targets=2,
                texts=train_texts,
                validation_texts=validation_texts,
                artifact_path=artifact,
                batch_size=batch_size,
                model_dtype=model_dtype,
            )
            row.update(
                {
                    "method": "lora",
                    "budget": None,
                    "rank": rank,
                    "learning_rate": lr,
                    "validation_delta_vs_base": row["validation_loss"] - base["base_validation_loss"],
                }
            )
            rows.append(row)
    summary = {
        "base_validation_loss": base["base_validation_loss"],
        "best_saint": _best(rows, "saint"),
        "best_lora": _best(rows, "lora"),
    }
    result = {
        "model_path": str(model_path),
        "corpus_path": str(corpus_path),
        "train_examples": len(train_texts),
        "validation_examples": len(validation_texts),
        "steps": steps,
        "batch_size": batch_size,
        "base": base,
        "rows": rows,
        "summary": summary,
        "generation": generation,
    }
    (root / "grid_results.json").write_text(dumps(result, indent=2), encoding="utf-8")
    (root / "grid_results.md").write_text(_markdown(rows, summary), encoding="utf-8")
    return result


__all__ = ["run_hf_phase13_grid"]
