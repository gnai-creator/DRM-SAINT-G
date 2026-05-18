"""Probe target matrix placement for Phase 15 14B experiments."""

from __future__ import annotations

import argparse
from json import dumps
from pathlib import Path

from saint.adapters.huggingface_loading import load_causal_lm


def _device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _metadata(args) -> dict[str, str]:
    return {
        key: value
        for key, value in {
            "model_dtype": args.model_dtype,
            "hf_device_map": args.hf_device_map,
            "hf_max_memory": args.hf_max_memory,
            "hf_offload_folder": args.hf_offload_folder,
        }.items()
        if value
    }


def run(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    device = _device(torch, args.device)
    model = load_causal_lm(AutoModelForCausalLM, args.model, device, _metadata(args))
    requested = [item.strip() for item in args.target_names.split(",") if item.strip()]
    params = dict(model.named_parameters())
    targets = []
    for name in requested:
        param = params.get(name)
        targets.append(
            {
                "name": name,
                "exists": param is not None,
                "device": str(param.device) if param is not None else None,
                "shape": list(param.shape) if param is not None else None,
                "numel": int(param.numel()) if param is not None else 0,
            }
        )
    cuda_targets = [
        item for item in targets if str(item.get("device", "")).startswith("cuda")
    ]
    result = {
        "model": args.model,
        "hf_device_map": args.hf_device_map,
        "hf_max_memory": args.hf_max_memory,
        "targets": targets,
        "cuda_target_count": len(cuda_targets),
        "device_map": {
            str(key): str(value)
            for key, value in getattr(model, "hf_device_map", {}).items()
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="runs/phase15_target_probe.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--hf-device-map", default=None)
    parser.add_argument("--hf-max-memory", default=None)
    parser.add_argument("--hf-offload-folder", default=None)
    parser.add_argument(
        "--target-names",
        default="model.layers.0.self_attn.q_proj.weight",
    )
    args = parser.parse_args()
    print(dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
