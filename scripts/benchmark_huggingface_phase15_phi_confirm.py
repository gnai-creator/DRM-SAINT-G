"""Confirm Phase 15 SAINT Phi results across seeds, layers, and ranks."""

from __future__ import annotations

import argparse
from copy import copy
from json import dumps
from pathlib import Path
from typing import Any

from benchmark_huggingface_phase15_compare import (
    _ints,
    _items,
    _lora_rank,
    _memory_items,
    _row_from_saint,
    _run_saint_subprocess,
)
from benchmark_huggingface_phase15_phi import MARCO12_GAIN


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def _gain(row: dict[str, Any]) -> float:
    return float(row.get("validation_gain_per_parameter") or 0.0)


def _delta(row: dict[str, Any]) -> float | None:
    value = row.get("validation_loss_delta")
    return float(value) if value is not None else None


def _target(layer: int, suffix: str) -> str:
    return f"model.layers.{layer}.self_attn.{suffix}.weight"


def _phi_case(args, *, seed: int, layer: int, rank: int, memory: str) -> dict[str, Any]:
    target = _target(layer, args.target_suffix)
    namespace = copy(args)
    namespace.seed = seed
    namespace.target_names = target
    namespace.routing_method = args.phi_routing_method
    namespace.phi_variant = args.phi_variant
    namespace.phi_rank = rank
    namespace.out = str(
        Path(args.out) / f"phi_s{seed}_l{layer}_r{rank}_{_safe(memory)}"
    )
    result = _run_saint_subprocess(namespace, budget=args.budget, max_memory=memory)
    row = _row_from_saint(result, budget=args.budget, max_memory=memory)
    row.update(
        {
            "method": "saint_phi_delta",
            "seed": seed,
            "layer": layer,
            "target": target,
            "phi_rank": rank,
            "phi_variant": args.phi_variant,
            "phi_source": args.phi_source,
            "beats_marco12_gain_per_parameter": _gain(row) >= MARCO12_GAIN,
        }
    )
    return row


def _lora_case(args, *, seed: int, layer: int) -> dict[str, Any]:
    namespace = copy(args)
    namespace.seed = seed
    namespace.target_names = _target(layer, args.target_suffix)
    namespace.out = str(Path(args.out) / f"lora_s{seed}_l{layer}")
    row = _lora_rank(namespace, rank=1)
    row.update({"seed": seed, "layer": layer, "target": namespace.target_names})
    return row


def _aggregate(rows: list[dict[str, Any]], *, method: str) -> dict[str, Any]:
    subset = [row for row in rows if row.get("method") == method and row.get("status") == "ok"]
    if not subset:
        return {"count": 0}
    gains = [_gain(row) for row in subset]
    deltas = [_delta(row) for row in subset if _delta(row) is not None]
    return {
        "count": len(subset),
        "mean_validation_gain_per_parameter": sum(gains) / len(gains),
        "best_validation_gain_per_parameter": max(gains),
        "mean_validation_delta": sum(deltas) / len(deltas) if deltas else None,
        "wins_vs_marco12": sum(1 for row in subset if _gain(row) >= MARCO12_GAIN),
    }


def _best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    subset = [row for row in rows if row.get("method") == "saint_phi_delta" and row.get("status") == "ok"]
    return max(subset, key=_gain) if subset else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    phi = _aggregate(rows, method="saint_phi_delta")
    lora = _aggregate(rows, method="lora_rank1_train_only")
    return {
        "best_phi": _best(rows),
        "phi": phi,
        "lora_rank1": lora,
        "passed": phi.get("best_validation_gain_per_parameter", 0.0) >= MARCO12_GAIN,
        "criterion": "Phi hadamard must reproduce or beat Marco 12 on at least one confirmed run",
    }


def _markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 15 Marco 14 Phi Confirmation",
        "",
        "| method | seed | layer | rank | val delta | val gain/param | params | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {seed} | {layer} | {rank} | {delta} | {gain} | {params} | {status} |".format(
                method=row.get("method", ""),
                seed=row.get("seed", ""),
                layer=row.get("layer", ""),
                rank=row.get("phi_rank") or row.get("rank", ""),
                delta="" if _delta(row) is None else f"{_delta(row):.6f}",
                gain=f"{_gain(row):.6e}",
                params="" if row.get("parameter_count") is None else row.get("parameter_count"),
                status=row.get("status", ""),
            )
        )
    lines.extend(["", "```json", dumps(summary, indent=2), "```"])
    return "\n".join(lines) + "\n"


def run(args) -> dict[str, Any]:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    memory = _memory_items(args.max_memories)[0]
    for seed in _ints(args.seeds):
        for layer in _ints(args.layers):
            for rank in _ints(args.phi_ranks):
                rows.append(_phi_case(args, seed=seed, layer=layer, rank=rank, memory=memory))
    if not args.skip_lora:
        for seed in _ints(args.lora_seeds):
            for layer in _ints(args.layers):
                try:
                    rows.append(_lora_case(args, seed=seed, layer=layer))
                except Exception as exc:  # pragma: no cover - large-model diagnostic.
                    rows.append(
                        {
                            "method": "lora_rank1_train_only",
                            "seed": seed,
                            "layer": layer,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
    summary = _summary(rows)
    result = {"model": args.model, "rows": rows, "summary": summary}
    (root / "phase15_phi_confirm_results.json").write_text(
        dumps(result, indent=2),
        encoding="utf-8",
    )
    (root / "phase15_phi_confirm_results.md").write_text(
        _markdown(rows, summary),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", default="data/tinyshakespeare_phase13.txt")
    parser.add_argument("--out", default="runs/phase15_marco14_phi_confirm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--seeds", default="31,32,33")
    parser.add_argument("--layers", default="1,2,3")
    parser.add_argument("--target-suffix", default="v_proj")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--budgets", default="32")
    parser.add_argument("--max-memories", default="0=14GiB,cpu=64GiB")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-texts", type=int, default=3)
    parser.add_argument("--validation-texts", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--routing-method", default="activation_phi_validation_rerank")
    parser.add_argument("--phi-routing-method", default="activation_phi_validation_rerank")
    parser.add_argument("--routing-max-length", type=int, default=4)
    parser.add_argument("--routing-batch-size", type=int, default=1)
    parser.add_argument("--routing-block-size", type=int, default=4)
    parser.add_argument("--target-names", default="model.layers.1.self_attn.v_proj.weight")
    parser.add_argument("--target-device", default="cuda")
    parser.add_argument("--max-cuda-gb", type=float, default=23.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--validate-during-train", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--validation-rerank-multiplier", type=int, default=4)
    parser.add_argument("--validation-rerank-chunk-size", type=int, default=256)
    parser.add_argument("--validation-probe-epsilon", type=float, default=1e-3)
    parser.add_argument("--validation-rerank-max-candidates", type=int, default=32)
    parser.add_argument("--validation-rerank-batch-size", type=int, default=8)
    parser.add_argument("--structured-prototype-count", type=int, default=1)
    parser.add_argument("--structured-prototype-mode", default="weight_sign")
    parser.add_argument("--structured-scale-granularity", default="block")
    parser.add_argument("--phi-ranks", default="2,4,8")
    parser.add_argument("--phi-rank", type=int, default=4)
    parser.add_argument("--phi-variant", default="hadamard")
    parser.add_argument("--phi-source", default="weight")
    parser.add_argument("--hf-device-map", default="auto")
    parser.add_argument("--lora-max-memory", default="0=14GiB,cpu=64GiB")
    parser.add_argument("--lora-learning-rate", type=float, default=0.001)
    parser.add_argument("--lora-ranks", default="1")
    parser.add_argument("--lora-seeds", default="31")
    parser.add_argument("--lora-b-init-scale", type=float, default=0.0)
    parser.add_argument("--skip-lora", action="store_true")
    args = parser.parse_args()
    print(dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
