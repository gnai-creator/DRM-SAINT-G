"""Small DRM-G sweep for graft checkpoints and internal target comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from saint.adapters.drm_grafting import run_drm_graft
from saint.config import RuntimeConfig


def _run(seed: int, target: str, init: str, batch_size: int) -> dict:
    config = RuntimeConfig(
        task="drm_transformer",
        method="drm_g_saint_phi_graft",
        seed=seed,
        steps=2,
        parameter_budget=64,
        metadata={
            "baseline_config": "configs/baselines/small_3.5M.yaml",
            "batch_size": batch_size,
            "seq_len": 8,
            "validation_seed": 1991 + seed,
            "phi_rank": 8,
            "projection_init": init,
            "target_module": target,
            "graft_scale": 1.0,
            "learning_rate": 0.005,
            "device": "cpu",
            "marco": "drm_g_marco_3_sweep",
        },
    )
    result = run_drm_graft(config)
    meta = result.metadata
    return {
        "seed": seed,
        "target_module": target,
        "projection_init": init,
        "batch_size": batch_size,
        "base_loss": meta["base_loss"],
        "graft_loss": result.test_loss,
        "validation_gain": meta["validation_gain"],
        "validation_gain_per_parameter": meta["validation_gain_per_parameter"],
        "dense_budget_gain": meta["dense_budget_gain"],
        "beats_dense": meta["validation_gain"] > meta["dense_budget_gain"],
        "decision": meta.get("graft_decision_status", "unknown"),
    }


def _markdown(rows: list[dict]) -> str:
    lines = [
        "# DRM-G Marco 3 Sweep",
        "",
        "| seed | target | init | batch | val_gain | dense_gain | beats_dense | decision |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | `{target_module}` | `{projection_init}` | {batch_size} | "
            "{validation_gain:.6f} | {dense_budget_gain:.6f} | {beats_dense} | "
            "{decision} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/drm_g_marco3_sweep")
    args = parser.parse_args()
    targets = ["blocks.0", "blocks.1", "blocks.2", "final_norm"]
    inits = ["activation", "gradient"]
    seeds = [31, 32, 33]
    rows = [
        _run(seed, target, init, batch_size=2)
        for seed in seeds
        for target in targets
        for init in inits
    ]
    rows.sort(key=lambda row: row["validation_gain_per_parameter"], reverse=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "results.md").write_text(_markdown(rows), encoding="utf-8")
    print(json.dumps({"best": rows[0], "count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
