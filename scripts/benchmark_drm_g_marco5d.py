"""DRM-G Marco 5D benchmark on a second DRM size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.benchmark_drm_g_marco5c_phi_variants import (
    _base_loss,
    _train_full_module,
    _train_variant,
)
from saint.adapters.drm_grafting import _import_drm, load_drm_baseline_config


SECOND_SIZE_CONFIG = "configs/scaling/multilingual/5m.yaml"


def _metadata(seed: int, validation_batches: int, batch_size: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "baseline_config": SECOND_SIZE_CONFIG,
        "device": "cpu",
        "batch_size": batch_size,
        "seq_len": 8,
        "learning_rate": 0.005,
        "use_real_tokens": True,
        "real_data_dir": "data/baseline",
        "validation_split": "val",
        "validation_batches": validation_batches,
        "data_seed": seed,
        "validation_seed": 4000 + seed,
    }


def _variants(d_model: int, steps: int) -> list[dict[str, Any]]:
    return [
        {"name": "phi_zero_full_rank", "rank": d_model, "init": "zero", "steps": steps},
        {
            "name": "phi_ls_full_rank",
            "rank": d_model,
            "init": "least_squares_gradient",
            "steps": steps,
        },
        {
            "name": "phi_ls_residual_full_rank",
            "rank": d_model - 1,
            "init": "least_squares_gradient",
            "residual_k": (d_model * 2) - 1,
            "steps": steps,
        },
        {
            "name": "phi_ls_train_ab_half_rank",
            "rank": d_model // 2,
            "init": "least_squares_gradient",
            "train_ab": True,
            "steps": steps,
            "lr": 0.002,
        },
    ]


def _run_seed(seed: int, args) -> list[dict[str, Any]]:
    metadata = _metadata(seed, args.validation_batches, args.batch_size)
    torch, config_cls, model_cls, _drm_root = _import_drm(metadata)
    drm_config = load_drm_baseline_config(metadata, config_cls)
    base = _base_loss(torch, model_cls, drm_config, metadata)
    rows = []
    for variant in _variants(int(drm_config.d_model), args.steps):
        row = _train_variant(torch, model_cls, drm_config, metadata, variant)
        row["base_loss"] = base
        row["validation_gain"] = base - row["final_loss"]
        row["gain_per_parameter"] = row["validation_gain"] / max(1, row["trainable_parameters"])
        row["d_model"] = int(drm_config.d_model)
        row["d_ff"] = int(drm_config.d_ff)
        rows.append(row)
    full = _train_full_module(torch, model_cls, drm_config, metadata, args.steps)
    full["base_loss"] = base
    full["validation_gain"] = base - full["final_loss"]
    full["gain_per_parameter"] = full["validation_gain"] / max(1, full["trainable_parameters"])
    full["d_model"] = int(drm_config.d_model)
    full["d_ff"] = int(drm_config.d_ff)
    rows.append(full)
    return rows


def _method_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        summaries.append({
            "method": method,
            "mean_gain": sum(row["validation_gain"] for row in subset) / len(subset),
            "mean_gain_per_parameter": sum(row["gain_per_parameter"] for row in subset) / len(subset),
            "positive_runs": sum(1 for row in subset if row["validation_gain"] > 0.0),
            "run_count": len(subset),
            "params": int(subset[0]["trainable_parameters"]),
        })
    summaries.sort(key=lambda row: row["mean_gain"], reverse=True)
    return summaries


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = _method_summary(rows)
    best = max(rows, key=lambda row: row["validation_gain"])
    best_phi = max(
        [row for row in rows if row["method"] != "full_module_linear"],
        key=lambda row: row["validation_gain"],
    )
    best_full = max(
        [row for row in rows if row["method"] == "full_module_linear"],
        key=lambda row: row["validation_gain"],
    )
    return {
        "second_size_config": SECOND_SIZE_CONFIG,
        "row_count": len(rows),
        "method_summary": summaries,
        "best": best,
        "best_phi": best_phi,
        "best_full_module": best_full,
        "phi_mean_beats_full": summaries[0]["method"] != "full_module_linear",
        "phase_5d_passed": any(row["positive_runs"] > 0 for row in summaries),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# DRM-G Marco 5D Second Size",
        "",
        f"- phase_5d_passed: {summary['phase_5d_passed']}",
        f"- phi_mean_beats_full: {summary['phi_mean_beats_full']}",
        "",
        "| method | mean_gain | mean_gain/param | positive | params |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["method_summary"]:
        lines.append(
            "| {method} | {mean_gain:.6f} | {mean_gain_per_parameter:.6e} | "
            "{positive_runs}/{run_count} | {params} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/drm_g_marco5d_second_size")
    parser.add_argument("--seeds", nargs="*", type=int, default=[31, 32, 33, 34])
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        rows.extend(_run_seed(seed, args))
    summary = _summary(rows)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
