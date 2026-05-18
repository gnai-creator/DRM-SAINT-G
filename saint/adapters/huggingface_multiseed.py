"""Phase 13 Marco 9 multiseed benchmark."""

from __future__ import annotations

from json import dumps
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from saint.adapters.huggingface_grid import run_hf_phase13_grid
from saint.adapters.huggingface_lora import (
    evaluate_lora_artifact,
    generate_with_lora_artifact,
)
from saint.adapters.huggingface_validation import load_text_corpus
from saint.adapters.huggingface_validation import _generate


def prepare_external_corpus(
    path: str | Path,
    *,
    url: str | None = None,
    max_lines: int = 96,
) -> Path:
    target = Path(path)
    if not target.exists():
        if not url:
            raise ValueError(f"dataset does not exist and no URL was provided: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = urlopen(url, timeout=30).read().decode("utf-8", errors="ignore")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        target.write_text("\n".join(lines[:max_lines]) + "\n", encoding="utf-8")
    return target


def _aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    subset = [row for row in rows if row["method"] == method]
    if not subset:
        return {}
    validation_losses = [row["validation_loss"] for row in subset]
    gains = [row["gain_per_parameter"] for row in subset]
    return {
        "count": len(subset),
        "mean_validation_loss": sum(validation_losses) / len(validation_losses),
        "best_validation_loss": min(validation_losses),
        "mean_gain_per_parameter": sum(gains) / len(gains),
    }


def _generation_metrics(text: str, prompt: str) -> dict[str, Any]:
    tokens = text.split()
    return {
        "changed_from_prompt": text.strip() != prompt.strip(),
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
    }


def _find_lora_artifact(run_dir: Path, best_lora: dict[str, Any]) -> Path:
    rank = int(best_lora["rank"])
    lr = str(best_lora["learning_rate"]).replace(".", "p").replace("-", "m")
    path = run_dir / f"lora_r{rank}_lr{lr}.pt"
    if not path.exists():
        raise ValueError(f"missing LoRA artifact: {path}")
    return path


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "| method | count | mean val loss | best val loss | mean gain/param |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, item in result["aggregate"].items():
        lines.append(
            "| {method} | {count} | {mean:.6f} | {best:.6f} | {gain:.8f} |".format(
                method=method,
                count=item["count"],
                mean=item["mean_validation_loss"],
                best=item["best_validation_loss"],
                gain=item["mean_gain_per_parameter"],
            )
        )
    lines.append("")
    lines.append("Decision: " + result["phase13_decision"])
    return "\n".join(lines) + "\n"


def run_hf_phase13_multiseed(
    model_path: str | Path,
    corpus_path: str | Path,
    run_dir: str | Path,
    *,
    dataset_url: str | None = None,
    seeds: tuple[int, ...] = (31, 32, 33),
    steps: int = 4,
    saint_budgets: tuple[int, ...] = (8, 16),
    saint_lrs: tuple[float, ...] = (0.001, 0.005),
    lora_ranks: tuple[int, ...] = (2, 4),
    lora_lrs: tuple[float, ...] = (0.001, 0.005),
    device: str = "auto",
    max_length: int = 32,
    batch_size: int = 4,
    saint_target_matrices: int = 2,
    saint_routing_method: str = "gradient",
    saint_routing_max_length: int | None = None,
    saint_routing_batch_size: int | None = None,
    prompts: tuple[str, ...] = ("SAINT", "Checkpoint", "Training"),
) -> dict[str, Any]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    corpus = prepare_external_corpus(corpus_path, url=dataset_url)
    all_rows: list[dict[str, Any]] = []
    seed_results = []
    best_lora_artifact: Path | None = None
    corpus_texts = load_text_corpus(corpus)
    validation_texts = corpus_texts[-max(1, len(corpus_texts) // 4):]
    for seed in seeds:
        seed_dir = root / f"seed_{seed}"
        result = run_hf_phase13_grid(
            model_path,
            corpus,
            seed_dir,
            seed=seed,
            steps=steps,
            saint_budgets=saint_budgets,
            saint_lrs=saint_lrs,
            lora_ranks=lora_ranks,
            lora_lrs=lora_lrs,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            saint_target_matrices=saint_target_matrices,
            saint_routing_method=saint_routing_method,
            saint_routing_max_length=saint_routing_max_length,
            saint_routing_batch_size=saint_routing_batch_size,
            prompts=prompts,
        )
        seed_results.append({"seed": seed, "summary": result["summary"]})
        all_rows.extend({**row, "seed": seed} for row in result["rows"])
        if best_lora_artifact is None and result["summary"]["best_lora"]:
            best_lora_artifact = _find_lora_artifact(seed_dir, result["summary"]["best_lora"])
    aggregate = {
        "saint": _aggregate(all_rows, "saint"),
        "lora": _aggregate(all_rows, "lora"),
    }
    best_lora_eval = (
        evaluate_lora_artifact(
            model_path,
            best_lora_artifact,
            validation_texts,
            device_name=device,
            max_length=max_length,
        )
        if best_lora_artifact is not None
        else {}
    )
    generation = {}
    if best_lora_artifact is not None:
        for prompt in prompts:
            base_text = _generate(model_path, prompt=prompt, device_name=device)
            text = generate_with_lora_artifact(
                model_path,
                best_lora_artifact,
                prompt=prompt,
                device_name=device,
            )
            generation[prompt] = {
                "base": base_text,
                "lora_loaded": text,
                "metrics": {
                    **_generation_metrics(text, prompt),
                    "changed_from_base": text != base_text,
                },
            }
    saint_mean = aggregate["saint"].get("mean_validation_loss", float("inf"))
    lora_mean = aggregate["lora"].get("mean_validation_loss", float("inf"))
    decision = (
        "fase_13_can_close_with_caveat"
        if saint_mean <= lora_mean
        else "needs_larger_model_or_dataset_before_close"
    )
    result = {
        "model_path": str(model_path),
        "corpus_path": str(corpus),
        "dataset_url": dataset_url,
        "seeds": list(seeds),
        "steps": steps,
        "rows": all_rows,
        "seed_results": seed_results,
        "aggregate": aggregate,
        "lora_loaded_eval": best_lora_eval,
        "generation": generation,
        "phase13_decision": decision,
    }
    (root / "multiseed_results.json").write_text(dumps(result, indent=2), encoding="utf-8")
    (root / "multiseed_results.md").write_text(_markdown(result), encoding="utf-8")
    return result


__all__ = ["prepare_external_corpus", "run_hf_phase13_multiseed"]
