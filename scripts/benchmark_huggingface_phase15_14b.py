"""Run a controlled Phase 15 14B load/forward smoke."""

from __future__ import annotations

import argparse
from json import dumps
from pathlib import Path
from time import perf_counter

from saint.adapters.huggingface_loading import load_causal_lm, model_dtype


def _device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _token_device(torch, model, fallback):
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return fallback


def _cuda_bytes(torch, device) -> int:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _device(torch, args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    metadata = _metadata(args)
    dtype = model_dtype(torch, args.model_dtype)
    model = load_causal_lm(AutoModelForCausalLM, args.model, device, metadata)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    load_s = perf_counter() - start
    load_cuda_peak = _cuda_bytes(torch, device)
    token_device = _token_device(torch, model, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    encoded = {key: value.to(token_device) for key, value in encoded.items()}
    forward_start = perf_counter()
    with torch.no_grad():
        output = model(**encoded, labels=encoded["input_ids"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_s = perf_counter() - forward_start
    forward_cuda_peak = _cuda_bytes(torch, device)
    result = {
        "model": args.model,
        "device": str(device),
        "token_device": str(token_device),
        "model_dtype": str(dtype).replace("torch.", "") if dtype is not None else "default",
        "hf_device_map": args.hf_device_map,
        "hf_max_memory": args.hf_max_memory,
        "load_s": load_s,
        "forward_s": forward_s,
        "loss": float(output.loss.detach().cpu().item()),
        "load_cuda_peak_bytes": load_cuda_peak,
        "forward_cuda_peak_bytes": forward_cuda_peak,
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
    parser.add_argument("--out", default="runs/phase15_marco1_14b_smoke.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--hf-device-map", default=None)
    parser.add_argument("--hf-max-memory", default=None)
    parser.add_argument("--hf-offload-folder", default=None)
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--prompt", default="SAINT controlled 14B smoke")
    args = parser.parse_args()
    result = run(args)
    print(dumps(result, indent=2))


if __name__ == "__main__":
    main()
