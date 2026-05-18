"""Run Phase 13 Hugging Face SAINT sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

from saint.adapters.huggingface_sweep import run_hf_phase13_sweep


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="runs/phase13_hf_sweep")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--seeds", default="31")
    parser.add_argument("--saint-budgets", default="4,8,16")
    parser.add_argument("--lora-ranks", default="1,2,4,8")
    args = parser.parse_args()

    result = run_hf_phase13_sweep(
        args.model,
        args.out,
        seeds=_ints(args.seeds),
        saint_budgets=_ints(args.saint_budgets),
        lora_ranks=_ints(args.lora_ranks),
        steps=args.steps,
        device=args.device,
        max_length=args.max_length,
    )
    out = Path(args.out)
    print(f"rows={len(result['rows'])}")
    print(f"json={out / 'results.json'}")
    print(f"markdown={out / 'results.md'}")


if __name__ == "__main__":
    main()
