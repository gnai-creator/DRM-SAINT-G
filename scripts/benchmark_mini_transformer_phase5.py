"""Run the phase-5 mini-transformer benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saint.transformer import (
    evaluate_mini_transformer_closure,
    run_mini_transformer_sweep,
    summarize_mini_transformer_rows,
)


def _write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 5 Mini-Transformer Benchmark",
        "",
        "| Method | Runs | Avg Test Loss | Avg Params | Avg Gain/Param |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {method} | {runs} | {loss:.6f} | {params:.1f} | {gain:.8f} |".format(
                method=row["method"],
                runs=row["runs"],
                loss=row["avg_test_loss"],
                params=row["avg_parameter_count"],
                gain=row["avg_gain_per_parameter"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/phase5_mini_transformer")
    parser.add_argument("--seeds", default="31,32")
    parser.add_argument("--delta-modes", default="repeated,dense")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--parameter-budget", type=int, default=48)
    parser.add_argument("--delta-scale", type=float, default=3.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    delta_modes = tuple(mode.strip() for mode in args.delta_modes.split(",") if mode.strip())
    rows = run_mini_transformer_sweep(
        seeds=seeds,
        delta_modes=delta_modes,
        steps=args.steps,
        parameter_budget=args.parameter_budget,
        delta_scale=args.delta_scale,
    )
    summaries = summarize_mini_transformer_rows(rows)
    decision = evaluate_mini_transformer_closure(rows)

    rows_path = out_dir / "mini_transformer_rows.json"
    summary_path = out_dir / "mini_transformer_summary.json"
    md_path = out_dir / "mini_transformer.md"
    decisions_path = out_dir / "mini_transformer_decision.json"
    rows_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    decisions_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    _write_markdown(md_path, summaries)

    print(f"rows={len(rows)}")
    print(f"summary={summary_path}")
    print(f"decision={decisions_path}")
    print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
