"""Benchmark progressive DRM-G graft cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from saint.config import RuntimeConfig
from saint.runtime.runner import train_runtime


SEQUENCES = {
    "activation_stack": [
        {"target_module": "blocks.2", "projection_init": "activation"},
        {"target_module": "final_norm", "projection_init": "activation"},
        {"target_module": "blocks.1", "projection_init": "gradient"},
    ],
    "linear_stack": [
        {"target_module": "blocks.1.attn.out_proj", "projection_init": "gradient"},
        {"target_module": "blocks.2.attn.out_proj", "projection_init": "gradient"},
        {"target_module": "blocks.3.attn.out_proj", "projection_init": "gradient"},
    ],
}


def _metadata(seed: int, sequence: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "baseline_config": "configs/baselines/small_3.5M.yaml",
        "device": "cpu",
        "batch_size": 4,
        "seq_len": 8,
        "phi_rank": 8,
        "graft_scale": 1.0,
        "learning_rate": 0.005,
        "use_real_tokens": True,
        "real_data_dir": "data/baseline",
        "validation_split": "val",
        "validation_batches": 3,
        "data_seed": seed,
        "validation_seed": 2000 + seed,
        "old_validation_seed": 1000 + seed,
        "require_beats_dense": True,
        "min_validation_gain": 0.0,
        "min_gain_per_parameter": 0.0,
        "defer_gain_floor": -0.00005,
        "grafts": sequence,
    }


def _run(seed: int, name: str, sequence: list[dict[str, str]], out_dir: Path) -> dict[str, Any]:
    run_dir = out_dir / f"{name}_seed{seed}"
    config = RuntimeConfig(
        experiment_name=f"drm_g_marco4_{name}_seed{seed}",
        task="drm_transformer",
        method="drm_g_saint_phi_progressive",
        output_dir=str(run_dir),
        seed=seed,
        steps=2,
        parameter_budget=128,
        metadata=_metadata(seed, sequence),
    )
    manifest = train_runtime(config)
    meta = manifest["metadata"]
    checkpoint_bytes = sum(int(item.get("bytes", 0)) for item in manifest.get("files", []))
    return {
        "seed": seed,
        "sequence": name,
        "base_loss": meta["base_loss"],
        "final_loss": meta["final_loss"],
        "sequence_gain": meta["sequence_gain"],
        "gain_per_parameter": meta["sequence_gain_per_parameter"],
        "old_regression": meta["old_regression"],
        "approved_grafts": meta["approved_grafts"],
        "rejected_grafts": meta["rejected_grafts"],
        "deferred_grafts": meta["deferred_grafts"],
        "approval_rate": meta["approval_rate"],
        "conflict_count": meta["conflict_count"],
        "parameter_count": manifest["parameter_count"],
        "checkpoint_bytes": checkpoint_bytes,
        "cuda_peak_bytes": meta["cuda_peak_bytes"],
        "routing_s": meta["routing_s"],
        "train_s": meta["train_s"],
        "eval_s": meta["eval_s"],
        "rows": meta["progressive_rows"],
        "run_dir": str(run_dir),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["sequence_gain"] > 0.0]
    two_plus = [row for row in rows if row["approved_grafts"] >= 2]
    best = max(rows, key=lambda row: row["gain_per_parameter"])
    return {
        "run_count": len(rows),
        "positive_runs": len(positive),
        "two_plus_graft_runs": len(two_plus),
        "best": best,
        "phase_passed": bool(two_plus and best["sequence_gain"] > 0.0),
    }


def _markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# DRM-G Marco 4 Benchmark",
        "",
        f"- run_count: {summary['run_count']}",
        f"- positive_runs: {summary['positive_runs']}",
        f"- two_plus_graft_runs: {summary['two_plus_graft_runs']}",
        f"- phase_passed: {summary['phase_passed']}",
        "",
        "| seed | sequence | gain | gain/param | approved | rejected | deferred | old_reg | ckpt bytes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {sequence} | {sequence_gain:.6f} | {gain_per_parameter:.6e} | "
            "{approved_grafts} | {rejected_grafts} | {deferred_grafts} | "
            "{old_regression:.6f} | {checkpoint_bytes} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/drm_g_marco4_benchmark")
    parser.add_argument("--seeds", nargs="*", type=int, default=[31, 32, 33, 34])
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _run(seed, name, sequence, out_dir)
        for seed in args.seeds
        for name, sequence in SEQUENCES.items()
    ]
    rows.sort(key=lambda row: row["gain_per_parameter"], reverse=True)
    summary = _summary(rows)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(_markdown(rows, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
