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


def _delta_payload(deltas, masks, base_weights) -> dict[str, list[list[float]]]:
    payload = {
        name: [[0.0 for _ in row] for row in matrix]
        for name, matrix in base_weights.items()
    }
    for name, delta in deltas.items():
        if name not in payload:
            continue
        matrix = (delta.detach() * masks[name]).cpu().tolist()
        payload[name] = [[float(value) for value in row] for row in matrix]
    return payload


def run_hf_forward(config: RuntimeConfig) -> MiniTransformerResult:
    torch, functional_call, AutoModelForCausalLM, AutoTokenizer = _require_deps()
    start = perf_counter()
    metadata = _metadata(config)
    source = metadata.get("model_name_or_path")
    if not source:
        raise ValueError("hf_saint_forward_smoke requires metadata.model_name_or_path")
    device = _device(torch, metadata)
    model = AutoModelForCausalLM.from_pretrained(str(source), local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    from saint.adapters.huggingface import make_task

    task = make_task(config)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    input_ids, attention_mask = _load_batch(
        tokenizer,
        _texts(metadata),
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
    initial_loss = float(_loss(functional_call, model, _merged_params(model, deltas, masks), input_ids, attention_mask).detach().cpu().item())
    for _ in range(max(1, int(config.steps))):
        optimizer.zero_grad()
        loss = _loss(functional_call, model, _merged_params(model, deltas, masks), input_ids, attention_mask)
        loss.backward()
        optimizer.step()
    final_loss = float(_loss(functional_call, model, _merged_params(model, deltas, masks), input_ids, attention_mask).detach().cpu().item())
    parameter_count = int(sum(mask.sum().item() for mask in masks.values()))
    return MiniTransformerResult(
        name="hf_saint_forward_smoke",
        train_loss=final_loss,
        test_loss=final_loss,
        parameter_count=parameter_count,
        optimizer_state_values=parameter_count * 2,
        elapsed_s=perf_counter() - start,
        metadata={
            "delta_payload": _delta_payload(deltas, masks, task.base_weights),
            "adapter": "huggingface_causal_lm",
            "autograd": True,
            "real_forward": True,
            "device": str(device),
            "initial_loss": initial_loss,
            "perplexity": exp(min(final_loss, 20.0)),
            "target_matrices": names,
            "marco": "fase_13_marco_3",
        },
    )


__all__ = ["run_hf_forward"]
