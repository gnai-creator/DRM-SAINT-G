#!/usr/bin/env python
"""Run Phase 16 Marco 4N-C offline NTK-hybrid score ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from saint.adapters.drm_grafting_ntk_hybrid_ablation import (
    analyze_runs,
    render_markdown_report,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Completed Marco 4N-B run directory. Repeat for multiple seeds.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where 4N-C offline ablation artifacts will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows, summary = analyze_runs(args.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "ntk_hybrid_score_ablation_rows.json", rows)
    write_json(output_dir / "ntk_hybrid_score_ablation_summary.json", summary)
    write_csv(output_dir / "ntk_hybrid_score_ablation_table.csv", rows)
    (output_dir / "ntk_hybrid_score_ablation.md").write_text(
        render_markdown_report(summary, rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} ablation rows to {output_dir}")
    print(
        "mean_gain={gain:.9f} mean_grafts={grafts:.3f} seeds_ge5={seeds}".format(
            gain=float(summary.get("mean_accumulated_gain", 0.0) or 0.0),
            grafts=float(summary.get("mean_accepted_grafts", 0.0) or 0.0),
            seeds=",".join(summary.get("seeds_with_five_or_more_grafts", [])) or "none",
        )
    )
    print("recommendations=" + ",".join(summary.get("recommendations", [])))


if __name__ == "__main__":
    main()
