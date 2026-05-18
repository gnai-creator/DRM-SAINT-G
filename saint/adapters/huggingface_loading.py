"""Shared Hugging Face loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def model_dtype(torch, requested: str | None):
    value = str(requested or "").lower()
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    return None


def parse_max_memory(value: Any) -> dict[Any, str] | None:
    if not value:
        return None
    if isinstance(value, dict):
        parsed = {}
        for key, item in value.items():
            parsed[int(key) if str(key).isdigit() else str(key)] = str(item)
        return parsed
    entries = {}
    for part in str(value).split(","):
        if not part.strip() or "=" not in part:
            continue
        key, item = part.split("=", 1)
        key = key.strip()
        entries[int(key) if key.isdigit() else key] = item.strip()
    return entries or None


def load_kwargs(torch, metadata: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": True}
    dtype = model_dtype(torch, metadata.get("model_dtype"))
    if dtype is not None:
        kwargs["dtype"] = dtype
    device_map = metadata.get("hf_device_map")
    if device_map:
        kwargs["device_map"] = str(device_map)
        kwargs["low_cpu_mem_usage"] = bool(metadata.get("hf_low_cpu_mem_usage", True))
    max_memory = parse_max_memory(metadata.get("hf_max_memory"))
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    offload_folder = metadata.get("hf_offload_folder")
    if offload_folder:
        target = Path(str(offload_folder))
        target.mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = str(target)
        kwargs["offload_state_dict"] = bool(metadata.get("hf_offload_state_dict", True))
    return kwargs


def load_causal_lm(AutoModelForCausalLM, source: str | Path, device, metadata):
    import torch

    kwargs = load_kwargs(torch, metadata)
    model = AutoModelForCausalLM.from_pretrained(str(source), **kwargs)
    if "device_map" not in kwargs:
        model = model.to(device)
    return model


__all__ = ["load_causal_lm", "load_kwargs", "model_dtype", "parse_max_memory"]
