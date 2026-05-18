"""Real Transformers forward path for small Hugging Face SAINT experiments."""

from __future__ import annotations

from math import exp
from time import perf_counter
from typing import Any

from saint.config import RuntimeConfig
from saint.transformer.training import MiniTransformerResult


def _require_deps():
    try:
        import torch
        from torch.func import functional_call
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and transformers are required for hf_saint_forward_smoke."
        ) from exc
    return torch, functional_call, AutoModelForCausalLM, AutoTokenizer


def _metadata(config: RuntimeConfig) -> dict[str, Any]:
    return dict(config.metadata or {})


def _device(torch, metadata: dict[str, Any]):
    requested = str(metadata.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _texts(metadata: dict[str, Any]) -> list[str]:
    values = metadata.get("texts")
    if isinstance(values, list) and values:
        return [str(item) for item in values]
    return [
        "simple ai node training",
        "saint trains compact deltas",
        "small local causal language model",
    ]


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
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _load_batches(
    tokenizer,
    texts: list[str],
    *,
    max_length: int,
    device,
    batch_size: int,
):
    size = max(1, batch_size)
    return [
        _load_batch(
            tokenizer,
            texts[index:index + size],
            max_length=max_length,
            device=device,
        )
        for index in range(0, len(texts), size)
    ]


def _target_names(model, metadata: dict[str, Any]) -> list[str]:
    keywords = tuple(metadata.get("target_keywords", ["c_attn.weight", "c_proj.weight", "lm_head.weight"]))
    candidates = [
        name
        for name, param in model.named_parameters()
        if param.ndim == 2 and any(keyword in name for keyword in keywords)
    ]
    return candidates[: max(1, int(metadata.get("max_trainable_matrices", 2)))]


def _mask_for_param(torch, param, *, budget: int):
    flat = param.detach().abs().flatten()
    count = max(1, min(budget, flat.numel()))
    indices = torch.topk(flat, k=count).indices
    mask = torch.zeros_like(flat)
    mask[indices] = 1.0
    return mask.reshape_as(param)


def _build_deltas(torch, model, names: list[str], *, parameter_budget: int):
    named = dict(model.named_parameters())
    per_matrix = max(1, parameter_budget // max(1, len(names)))
    deltas = {}
    masks = {}
    for name in names:
        param = named[name]
        deltas[name] = torch.zeros_like(param, requires_grad=True)
        masks[name] = _mask_for_param(torch, param, budget=per_matrix)
    return deltas, masks


def _merged_params(model, deltas, masks):
    params = dict(model.named_parameters())
    for name, delta in deltas.items():
        params[name] = params[name] + (delta * masks[name])
    return params


def _loss(functional_call, model, params, input_ids, attention_mask):
    kwargs = {"input_ids": input_ids, "labels": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return functional_call(model, params, (), kwargs).loss


def _loss_value(functional_call, model, params, input_ids, attention_mask) -> float:
    return float(
        _loss(functional_call, model, params, input_ids, attention_mask)
        .detach()
        .cpu()
        .item()
    )


def _delta_payload(deltas, masks, base_weights) -> dict[str, Any]:
    sparse = {}
    shapes = {
        name: [len(matrix), len(matrix[0]) if matrix else 0]
        for name, matrix in base_weights.items()
    }
    for name, delta in deltas.items():
        if name not in base_weights:
            continue
        rows = len(base_weights[name])
        cols = len(base_weights[name][0]) if rows else 0
        matrix = (delta.detach() * masks[name])[:rows, :cols].cpu()
        entries = []
        for row, col in matrix.nonzero(as_tuple=False).tolist():
            entries.append([row, col, float(matrix[row, col].item())])
        if entries:
            sparse[name] = entries
    return {"format": "saint_sparse_delta", "shapes": shapes, "values": sparse}


def run_hf_forward(config: RuntimeConfig) -> MiniTransformerResult:
    torch, functional_call, AutoModelForCausalLM, AutoTokenizer = _require_deps()
    start = perf_counter()
    metadata = _metadata(config)
    torch.manual_seed(int(config.seed))
    source = metadata.get("model_name_or_path")
    if not source:
        raise ValueError("hf_saint_forward_smoke requires metadata.model_name_or_path")
    device = _device(torch, metadata)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(str(source), local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    load_cuda_peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    from saint.adapters.huggingface import matrices_from_state

    base_weights = matrices_from_state(dict(model.state_dict()), metadata)
    if not base_weights:
        raise ValueError("no matching 2D Hugging Face matrices found")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    train_texts = _texts(metadata)
    input_ids, attention_mask = _load_batch(
        tokenizer,
        train_texts,
        max_length=int(metadata.get("max_length", 32)),
        device=device,
    )
    train_batches = _load_batches(
        tokenizer,
        train_texts,
        max_length=int(metadata.get("max_length", 32)),
        device=device,
        batch_size=int(metadata.get("batch_size", len(train_texts))),
    )
    validation_texts = metadata.get("validation_texts")
    val_ids, val_mask = _load_batch(
        tokenizer,
        [str(item) for item in validation_texts] if isinstance(validation_texts, list) else _texts(metadata),
        max_length=int(metadata.get("max_length", 32)),
        device=device,
    )
    names = _target_names(model, metadata)
    deltas, masks = _build_deltas(
        torch,
        model,
        names,
        parameter_budget=max(1, config.parameter_budget),
    )
    optimizer = torch.optim.AdamW(list(deltas.values()), lr=float(metadata.get("learning_rate", 1e-3)))
    initial_loss = _loss_value(functional_call, model, _merged_params(model, deltas, masks), input_ids, attention_mask)
    steps = max(1, int(config.steps))
    train_start = perf_counter()
    for _ in range(steps):
        optimizer.zero_grad()
        for batch_ids, batch_mask in train_batches:
            loss = _loss(
                functional_call,
                model,
                _merged_params(model, deltas, masks),
                batch_ids,
                batch_mask,
            ) / len(train_batches)
            loss.backward()
        optimizer.step()
    train_elapsed = max(perf_counter() - train_start, 1e-9)
    train_cuda_peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    final_loss = _loss_value(functional_call, model, _merged_params(model, deltas, masks), input_ids, attention_mask)
    validation_loss = _loss_value(functional_call, model, _merged_params(model, deltas, masks), val_ids, val_mask)
    parameter_count = int(sum(mask.sum().item() for mask in masks.values()))
    tokens_seen = sum(int(ids.numel()) for ids, _ in train_batches) * steps
    cuda_peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    return MiniTransformerResult(
        name="hf_saint_forward_smoke",
        train_loss=final_loss,
        test_loss=final_loss,
        parameter_count=parameter_count,
        optimizer_state_values=parameter_count * 2,
        elapsed_s=perf_counter() - start,
        metadata={
            "delta_payload": _delta_payload(deltas, masks, base_weights),
            "adapter": "huggingface_causal_lm",
            "autograd": True,
            "real_forward": True,
            "device": str(device),
            "initial_loss": initial_loss,
            "perplexity": exp(min(final_loss, 20.0)),
            "validation_loss": validation_loss,
            "validation_perplexity": exp(min(validation_loss, 20.0)),
            "tokens_per_s": tokens_seen / train_elapsed,
            "tokens_seen": tokens_seen,
            "batch_count": len(train_batches),
            "cuda_peak_bytes": cuda_peak,
            "load_cuda_peak_bytes": load_cuda_peak,
            "train_cuda_peak_bytes": train_cuda_peak,
            "delta_payload_format": "saint_sparse_delta",
            "target_matrices": names,
            "marco": "fase_13_marco_3",
        },
    )


__all__ = ["run_hf_forward"]
