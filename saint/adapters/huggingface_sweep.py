"""Sweep utilities for Phase 13 Hugging Face comparisons."""

from __future__ import annotations

from json import dumps
from math import exp
from pathlib import Path
from typing import Any

from saint.adapters.huggingface_benchmark import (
    _batch,
    _full_finetune,
    _gain_per_parameter,
    _loss,
    _lora_finetune,
    _require_deps,
)
from saint.config import RuntimeConfig


def _model_kind(model_path: str | Path) -> str:
    path = Path(model_path)
    if "tiny_model" in path.parts:
        return "synthetic_local_fixture"
    return "local_huggingface_model"


def _evaluate_merged(
    model_path: str | Path,
    merged_weights: dict[str, list[list[float]]],
    *,
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
    input_ids, attention_mask = _batch(tokenizer, device, max_length=max_length)
    loss = float(_loss(model, input_ids, attention_mask).detach().cpu().item())
    return {"merged_loss": loss, "merged_perplexity": exp(min(loss, 20.0))}


def _saint_row(
    model_path: str | Path,
    run_dir: Path,
    *,
    seed: int,
    steps: int,
    budget: int,
    learning_rate: float,
    device: str,
    max_length: int,
) -> dict[str, Any]:
    from saint.runtime import merge_runtime, resume_runtime, train_runtime

    saint_dir = run_dir / f"saint_budget_{budget}_seed_{seed}"
    config = RuntimeConfig(
        experiment_name=f"hf_saint_budget_{budget}_seed_{seed}",
        output_dir=str(saint_dir),
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
        },
    )
    result = train_runtime(config)
    resumed = resume_runtime(saint_dir)
    merged = merge_runtime(saint_dir)
    merged_eval = _evaluate_merged(
        model_path,
        merged["merged_weights"],
        device_name=result["metadata"]["device"],
        max_length=max_length,
    )
    initial = result["metadata"]["initial_loss"]
    final = result["train_loss"]
    return {
        "method": "hf_saint_forward_smoke",
        "seed": seed,
        "budget": budget,
        "rank": None,
        "initial_loss": initial,
        "train_loss": final,
        "loss_delta": final - initial,
        "initial_perplexity": exp(min(initial, 20.0)),
        "train_perplexity": exp(min(final, 20.0)),
        **merged_eval,
        "parameter_count": result["parameter_count"],
        "gain_per_parameter": _gain_per_parameter(
            initial,
            final,
            result["parameter_count"],
        ),
        "tokens_per_s": result["metadata"]["tokens_per_s"],
        "tokens_seen": result["metadata"]["tokens_seen"],
        "cuda_peak_bytes": result["metadata"]["cuda_peak_bytes"],
        "resume_quality_delta": abs(resumed["train_loss"] - final),
        "checkpoint_merge": bool(merged["merged"] and merged["shape_validation"]),
        "device": result["metadata"]["device"],
    }


def _format_markdown(rows: list[dict[str, Any]]) -> str:
    header = (
        "| method | seed | budget | rank | params | loss | merged ppl | "
        "gain/param | tokens/s | cuda peak |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {method} | {seed} | {budget} | {rank} | {params} | "
            "{loss:.6f} | {ppl:.6f} | {gain:.8f} | {tps:.2f} | {cuda} |".format(
                method=row["method"],
                seed=row["seed"],
                budget="" if row.get("budget") is None else row["budget"],
                rank="" if row.get("rank") is None else row["rank"],
                params=row["parameter_count"],
                loss=row["train_loss"],
                ppl=row.get("merged_perplexity", row.get("train_perplexity", 0.0)),
                gain=row["gain_per_parameter"],
                tps=row["tokens_per_s"],
                cuda=row["cuda_peak_bytes"],
            )
        )
    return "\n".join(lines) + "\n"


def run_hf_phase13_sweep(
    model_path: str | Path,
    run_dir: str | Path,
    *,
    seeds: tuple[int, ...] = (31,),
    saint_budgets: tuple[int, ...] = (4, 8, 16),
    lora_ranks: tuple[int, ...] = (1, 2, 4, 8),
    steps: int = 4,
    learning_rate: float = 1e-3,
    lora_learning_rate: float | None = None,
    device: str = "auto",
    max_length: int = 16,
) -> dict[str, Any]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for budget in saint_budgets:
            rows.append(
                _saint_row(
                    model_path,
                    root,
                    seed=seed,
                    steps=steps,
                    budget=budget,
                    learning_rate=learning_rate,
                    device=device,
                    max_length=max_length,
                )
            )
        for rank in lora_ranks:
            row = _lora_finetune(
                model_path,
                seed=seed,
                steps=steps,
                learning_rate=lora_learning_rate or learning_rate,
                device_name=device,
                max_length=max_length,
                rank=rank,
                alpha=float(rank),
                max_targets=2,
            )
            row.update(
                {
                    "budget": None,
                    "rank": rank,
                    "initial_perplexity": exp(min(row["initial_loss"], 20.0)),
                    "train_perplexity": exp(min(row["train_loss"], 20.0)),
                    "merged_perplexity": exp(min(row["train_loss"], 20.0)),
                    "resume_quality_delta": None,
                    "checkpoint_merge": None,
                }
            )
            rows.append(row)
        full = _full_finetune(
            model_path,
            seed=seed,
            steps=steps,
            learning_rate=learning_rate,
            device_name=device,
            max_length=max_length,
        )
        full.update(
            {
                "budget": None,
                "rank": None,
                "initial_perplexity": exp(min(full["initial_loss"], 20.0)),
                "train_perplexity": exp(min(full["train_loss"], 20.0)),
                "merged_perplexity": exp(min(full["train_loss"], 20.0)),
                "resume_quality_delta": None,
                "checkpoint_merge": None,
            }
        )
        rows.append(full)
    result = {
        "model_path": str(model_path),
        "model_kind": _model_kind(model_path),
        "seeds": list(seeds),
        "saint_budgets": list(saint_budgets),
        "lora_ranks": list(lora_ranks),
        "steps": steps,
        "device": device,
        "rows": rows,
    }
    (root / "results.json").write_text(dumps(result, indent=2), encoding="utf-8")
    (root / "results.md").write_text(_format_markdown(rows), encoding="utf-8")
    return result


__all__ = ["run_hf_phase13_sweep"]
