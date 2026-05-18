"""Run the phase-6 sensitivity-map benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saint.sensitivity import (
    evaluate_sensitivity_final,
    evaluate_sensitivity_success,
    run_sensitivity_sweep,
    summarize_sensitivity_by_regime,
    summarize_sensitivity_rows,
)


def _write_markdown(
    path: Path,
    summaries: list[dict],
    decision: dict,
    final_decision: dict,
) -> None:
    lines = [
        "# Phase 6 Sensitivity Benchmark",
        "",
        "| Method | Runs | Avg Test Loss | Avg Gain/Param |",
        "|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {method} | {runs} | {loss:.8f} | {gain:.10f} |".format(
                method=row["method"],
                runs=row["runs"],
                loss=row["avg_test_loss"],
                gain=row["avg_gain_per_parameter"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- passed: {decision['passed']}",
            f"- winner_count: {decision['winner_count']}",
            f"- reason: {decision['reason']}",
            "",
            "## Final Decision",
            "",
            f"- passed: {final_decision['passed']}",
            f"- regime_passes: {final_decision['regime_passes']}",
            f"- block_best_beats_random: {final_decision['block_best_beats_random']}",
            f"- accumulated_beats_random: {final_decision['accumulated_beats_random']}",
            "- sensitivity_saint_beats_default: "
            f"{final_decision['sensitivity_saint_beats_default']}",
            f"- reason: {final_decision['reason']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/phase6_sensitivity")
    parser.add_argument("--seeds", default="41,42")
    parser.add_argument("--delta-modes", default="repeated,dense")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--parameter-budget", type=int, default=48)
    parser.add_argument("--delta-scale", type=float, default=3.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    delta_modes = tuple(mode.strip() for mode in args.delta_modes.split(",") if mode.strip())
    rows = run_sensitivity_sweep(
        seeds=seeds,
        delta_modes=delta_modes,
        steps=args.steps,
        parameter_budget=args.parameter_budget,
        delta_scale=args.delta_scale,
    )
    summaries = summarize_sensitivity_rows(rows)
    regime_summaries = summarize_sensitivity_by_regime(rows)
    decision = evaluate_sensitivity_success(rows)
    final_decision = evaluate_sensitivity_final(rows)

    rows_path = out_dir / "sensitivity_rows.json"
    summary_path = out_dir / "sensitivity_summary.json"
    regime_summary_path = out_dir / "sensitivity_by_regime.json"
    decision_path = out_dir / "sensitivity_decision.json"
    final_decision_path = out_dir / "sensitivity_final_decision.json"
    md_path = out_dir / "sensitivity.md"
    rows_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    regime_summary_path.write_text(json.dumps(regime_summaries, indent=2), encoding="utf-8")
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    final_decision_path.write_text(json.dumps(final_decision, indent=2), encoding="utf-8")
    _write_markdown(md_path, summaries, decision, final_decision)

    print(f"rows={len(rows)}")
    print(f"summary={summary_path}")
    print(f"by_regime={regime_summary_path}")
    print(f"decision={decision_path}")
    print(f"final_decision={final_decision_path}")
    print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
