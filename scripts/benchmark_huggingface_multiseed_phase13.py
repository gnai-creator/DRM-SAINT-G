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
    parser.add_argument("--saint-target-names", default="")
    parser.add_argument("--saint-target-device", default=None)
    parser.add_argument("--saint-routing-method", default="gradient")
    parser.add_argument("--saint-routing-max-length", type=int, default=None)
    parser.add_argument("--saint-routing-batch-size", type=int, default=None)
    parser.add_argument("--model-dtype", default=None)
    parser.add_argument("--max-cuda-gb", type=float, default=None)
    parser.add_argument("--saint-delta-application", default="functional")
    parser.add_argument("--hf-device-map", default=None)
    parser.add_argument("--hf-max-memory", default=None)
    parser.add_argument("--hf-offload-folder", default=None)
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
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
        lora_ranks=() if args.skip_lora else _ints(args.lora_ranks),
        lora_lrs=() if args.skip_lora else _floats(args.lora_lrs),
        device=args.device,
        batch_size=args.batch_size,
        saint_target_matrices=args.saint_target_matrices,
        saint_target_names=tuple(
            item.strip() for item in args.saint_target_names.split(",") if item.strip()
        ),
        saint_target_device=args.saint_target_device,
        saint_routing_method=args.saint_routing_method,
        saint_routing_max_length=args.saint_routing_max_length,
        saint_routing_batch_size=args.saint_routing_batch_size,
        model_dtype=args.model_dtype,
        max_cuda_gb=args.max_cuda_gb,
        saint_delta_application=args.saint_delta_application,
        hf_load_metadata={
            key: value
            for key, value in {
                "hf_device_map": args.hf_device_map,
                "hf_max_memory": args.hf_max_memory,
                "hf_offload_folder": args.hf_offload_folder,
            }.items()
            if value
        },
        prompts=() if args.skip_generation else ("SAINT", "Checkpoint", "Training"),
    )
    out = Path(args.out)
    print(f"rows={len(result['rows'])}")
    print(f"decision={result['phase13_decision']}")
    print(f"json={out / 'multiseed_results.json'}")
    print(f"markdown={out / 'multiseed_results.md'}")


if __name__ == "__main__":
    main()
