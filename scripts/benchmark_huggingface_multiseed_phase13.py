"""Run Phase 13 Marco 9 multiseed benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from saint.adapters.huggingface_multiseed import run_hf_phase13_multiseed


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", default="data/tinyshakespeare_phase13.txt")
    parser.add_argument("--dataset-url", default=None)
    parser.add_argument("--out", default="runs/phase13_marco9_multiseed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seeds", default="31,32,33")
    parser.add_argument("--saint-budgets", default="8,16")
    parser.add_argument("--saint-lrs", default="0.001,0.005")
    parser.add_argument("--lora-ranks", default="2,4")
    parser.add_argument("--lora-lrs", default="0.001,0.005")
    parser.add_argument("--saint-target-matrices", type=int, default=2)
    parser.add_argument("--saint-routing-method", default="gradient")
    args = parser.parse_args()

    result = run_hf_phase13_multiseed(
        args.model,
        args.corpus,
        args.out,
        dataset_url=args.dataset_url,
        seeds=_ints(args.seeds),
        steps=args.steps,
        saint_budgets=_ints(args.saint_budgets),
        saint_lrs=_floats(args.saint_lrs),
        lora_ranks=_ints(args.lora_ranks),
        lora_lrs=_floats(args.lora_lrs),
        device=args.device,
        batch_size=args.batch_size,
        saint_target_matrices=args.saint_target_matrices,
        saint_routing_method=args.saint_routing_method,
    )
    out = Path(args.out)
    print(f"rows={len(result['rows'])}")
    print(f"decision={result['phase13_decision']}")
    print(f"json={out / 'multiseed_results.json'}")
    print(f"markdown={out / 'multiseed_results.md'}")


if __name__ == "__main__":
    main()
