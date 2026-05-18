"""Low-memory LoRA training helpers for Hugging Face diagnostics."""

from __future__ import annotations

import gc
from time import perf_counter
from typing import Any

from saint.adapters.huggingface_loading import load_causal_lm, model_dtype


def _load_batch(tokenizer, texts: list[str], *, max_length: int, device):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    mask = encoded.get("attention_mask")
    return encoded["input_ids"].to(device), mask.to(device) if mask is not None else None


def _plain_loss(model, input_ids, attention_mask):
    kwargs = {"input_ids": input_ids, "labels": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return model(**kwargs).loss


def _module_name(param_name: str) -> str:
    if param_name.endswith(".weight"):
        return param_name[:-7]
    return param_name.rsplit(".", 1)[0]


def _lora_hook(torch, a, b):
    def hook(_module, inputs, output):
        if not inputs:
            return output
        hidden = inputs[0].to(dtype=b.dtype)
        update = hidden.matmul(b.t()).matmul(a.t())
        return output + update.to(dtype=output.dtype, device=output.device)

    return hook


def train_lora_rank(args: Any, *, rank: int, texts: list[str]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metadata = {
        "model_dtype": args.model_dtype,
        "hf_device_map": args.hf_device_map,
        "hf_max_memory": args.lora_max_memory,
        "hf_offload_folder": str(args.lora_offload_folder),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = load_causal_lm(AutoModelForCausalLM, args.model, device, metadata)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_ids, attention_mask = _load_batch(
        tokenizer,
        texts,
        max_length=args.max_length,
        device=device,
    )
    named = dict(model.named_parameters())
    target = args.lora_target
    if target not in named:
        raise ValueError(f"missing LoRA target: {target}")
    for param in model.parameters():
        param.requires_grad_(False)
    weight = named[target]
    rows, cols = weight.shape
    dtype = model_dtype(torch, args.model_dtype) or weight.dtype
    a = (torch.randn(rows, rank, device=weight.device, dtype=dtype) * 0.01)
    b = torch.randn(rank, cols, device=weight.device, dtype=dtype)
    b.mul_(args.lora_b_init_scale)
    a.requires_grad_()
    b.requires_grad_()
    module = model.get_submodule(_module_name(target))
    handle = module.register_forward_hook(_lora_hook(torch, a, b))
    optimizer = torch.optim.AdamW([a, b], lr=args.lora_learning_rate)
    model.train()
    handle.remove()
    with torch.no_grad():
        initial_loss = float(_plain_loss(model, input_ids, attention_mask).cpu().item())
    handle = module.register_forward_hook(_lora_hook(torch, a, b))
    start = perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = _plain_loss(model, input_ids, attention_mask)
        loss.backward()
        optimizer.step()
    elapsed = perf_counter() - start
    model.eval()
    with torch.no_grad():
        final_loss = float(_plain_loss(model, input_ids, attention_mask).cpu().item())
    handle.remove()
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    if peak / 1_000_000_000 > args.max_cuda_gb:
        raise RuntimeError(f"CUDA budget exceeded during lora: {peak / 1_000_000_000:.3f} GB")
    params = int(a.numel() + b.numel())
    result = {
        "method": f"lora_rank{rank}_train_only",
        "budget": None,
        "rank": rank,
        "max_memory": args.lora_max_memory,
        "status": "ok",
        "elapsed_s": elapsed,
        "initial_loss": initial_loss,
        "train_loss": final_loss,
        "loss_delta": final_loss - initial_loss,
        "parameter_count": params,
        "gain_per_parameter": max(initial_loss - final_loss, 0.0) / max(1, params),
        "train_cuda_gb": peak / 1_000_000_000,
        "lora_application": "forward_hook",
    }
    del model, tokenizer, input_ids, attention_mask, a, b, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


__all__ = ["train_lora_rank"]
