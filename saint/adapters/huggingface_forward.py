"""Real Transformers forward path for small Hugging Face SAINT experiments."""

from __future__ import annotations

from math import exp
from time import perf_counter
from typing import Any

from saint.adapters.huggingface_loading import load_causal_lm, model_dtype
from saint.adapters.huggingface_routing import build_routed_deltas, merged_params
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


def _dtype(torch, metadata: dict[str, Any]):
    return model_dtype(torch, metadata.get("model_dtype"))


def _check_cuda_budget(torch, device, metadata: dict[str, Any], stage: str) -> None:
    if device.type != "cuda" or "max_cuda_gb" not in metadata:
        return
    peak_gb = torch.cuda.max_memory_allocated(device) / 1_000_000_000
    if peak_gb > float(metadata["max_cuda_gb"]):
        raise RuntimeError(
            f"CUDA budget exceeded during {stage}: {peak_gb:.3f} GB"
        )


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
    default_keywords = [
        "c_attn.weight",
        "c_proj.weight",
        "q_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
        "lm_head.weight",
    ]
    keywords = tuple(metadata.get("target_keywords", default_keywords))
    candidates = [
        name
        for name, param in model.named_parameters()
        if param.ndim == 2 and any(keyword in name for keyword in keywords)
    ]
    return candidates[: max(1, int(metadata.get("max_trainable_matrices", 2)))]


def _loss(functional_call, model, params, input_ids, attention_mask):
    kwargs = {"input_ids": input_ids, "labels": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return functional_call(model, params, (), kwargs).loss


def _plain_loss(model, input_ids, attention_mask):
    kwargs = {"input_ids": input_ids, "labels": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return model(**kwargs).loss


def _loss_value(functional_call, model, params, input_ids, attention_mask) -> float:
    return float(
        _loss(functional_call, model, params, input_ids, attention_mask)
        .detach()
        .cpu()
        .item()
    )


def _inplace_delta(torch, named, deltas, coordinates, *, sign: float) -> None:
    with torch.no_grad():
        for name, delta in deltas.items():
            rows, cols = coordinates[name]
            named[name].index_put_((rows, cols), sign * delta, accumulate=True)


def _inplace_loss_value(torch, model, named, deltas, coordinates, input_ids, attention_mask):
    _inplace_delta(torch, named, deltas, coordinates, sign=1.0)
    try:
        with torch.no_grad():
            value = _plain_loss(model, input_ids, attention_mask)
            return float(value.detach().cpu().item())
    finally:
        _inplace_delta(torch, named, deltas, coordinates, sign=-1.0)


def _zero_target_grads(named, coordinates) -> None:
    for name in coordinates:
        grad = named[name].grad
        if grad is not None:
            grad.zero_()


def _train_inplace_sparse(
    torch,
    model,
    deltas,
    coordinates,
    train_batches,
    *,
    steps: int,
    learning_rate: float,
) -> None:
    named = dict(model.named_parameters())
    for name in coordinates:
        named[name].requires_grad_(True)
    for _ in range(steps):
        _zero_target_grads(named, coordinates)
        for batch_ids, batch_mask in train_batches:
            _inplace_delta(torch, named, deltas, coordinates, sign=1.0)
            try:
                loss = _plain_loss(model, batch_ids, batch_mask) / len(train_batches)
                loss.backward()
            finally:
                _inplace_delta(torch, named, deltas, coordinates, sign=-1.0)
        with torch.no_grad():
            for name, delta in deltas.items():
                rows, cols = coordinates[name]
                grad = named[name].grad
                if grad is not None:
                    delta.sub_(learning_rate * grad[rows, cols].to(delta.dtype))
        _zero_target_grads(named, coordinates)
    for name in coordinates:
        named[name].requires_grad_(False)


def _target_shapes(model, names: list[str]) -> dict[str, list[int]]:
    params = dict(model.named_parameters())
    return {name: [int(params[name].shape[0]), int(params[name].shape[1])] for name in names}


def _delta_payload(deltas, coordinates, shapes: dict[str, list[int]]) -> dict[str, Any]:
    sparse = {}
    for name, delta in deltas.items():
        if name not in shapes:
            continue
        rows, cols = int(shapes[name][0]), int(shapes[name][1])
        entries = []
        row_indices, col_indices = coordinates[name]
        for row, col, value in zip(
            row_indices.detach().cpu().tolist(),
            col_indices.detach().cpu().tolist(),
            delta.detach().cpu().tolist(),
        ):
            if row < rows and col < cols and abs(float(value)) > 0.0:
                entries.append([int(row), int(col), float(value)])
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
    dtype = _dtype(torch, metadata)
    model = load_causal_lm(AutoModelForCausalLM, source, device, metadata)
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    load_cuda_peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    _check_cuda_budget(torch, device, metadata, "load")
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
    routing_length = int(metadata.get("routing_max_length", metadata.get("max_length", 32)))
    routing_batch_size = int(metadata.get("routing_batch_size", len(train_texts)))
    routing_ids, routing_mask = _load_batch(
        tokenizer,
        train_texts[: max(1, routing_batch_size)],
        max_length=routing_length,
        device=device,
    )
    validation_texts = metadata.get("validation_texts")
    val_ids, val_mask = _load_batch(
        tokenizer,
        [str(item) for item in validation_texts] if isinstance(validation_texts, list) else _texts(metadata),
        max_length=int(metadata.get("max_length", 32)),
        device=device,
    )
    names = _target_names(model, metadata)
    if not names:
        raise ValueError("no matching 2D Hugging Face matrices found")
    target_shapes = _target_shapes(model, names)
    routing_method = str(metadata.get("routing_method", "gradient"))
    delta_application = str(metadata.get("delta_application", "functional"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    deltas, coordinates = build_routed_deltas(
        torch,
        functional_call,
        model,
        names,
        parameter_budget=max(1, config.parameter_budget),
        input_ids=routing_ids,
        attention_mask=routing_mask,
        routing_method=routing_method,
        loss_fn=_loss,
    )
    routing_cuda_peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    _check_cuda_budget(torch, device, metadata, "routing")
    learning_rate = float(metadata.get("learning_rate", 1e-3))
    named = dict(model.named_parameters())
    if delta_application == "inplace":
        initial_loss = _inplace_loss_value(
            torch, model, named, deltas, coordinates, input_ids, attention_mask
        )
    else:
        optimizer = torch.optim.AdamW(list(deltas.values()), lr=learning_rate)
        initial_loss = _loss_value(
            functional_call,
            model,
            merged_params(torch, model, deltas, coordinates),
            input_ids,
            attention_mask,
        )
    steps = max(1, int(config.steps))
    train_start = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if delta_application == "inplace":
        _train_inplace_sparse(
            torch,
            model,
            deltas,
            coordinates,
            train_batches,
            steps=steps,
            learning_rate=learning_rate,
        )
    else:
        for _ in range(steps):
            optimizer.zero_grad()
            for batch_ids, batch_mask in train_batches:
                loss = _loss(
                    functional_call,
                    model,
                    merged_params(torch, model, deltas, coordinates),
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
    _check_cuda_budget(torch, device, metadata, "train")
    if delta_application == "inplace":
        final_loss = _inplace_loss_value(
            torch, model, named, deltas, coordinates, input_ids, attention_mask
        )
        validation_loss = _inplace_loss_value(
            torch, model, named, deltas, coordinates, val_ids, val_mask
        )
    else:
        final_loss = _loss_value(
            functional_call,
            model,
            merged_params(torch, model, deltas, coordinates),
            input_ids,
            attention_mask,
        )
        validation_loss = _loss_value(
            functional_call,
            model,
            merged_params(torch, model, deltas, coordinates),
            val_ids,
            val_mask,
        )
    parameter_count = int(sum(delta.numel() for delta in deltas.values()))
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
            "delta_payload": _delta_payload(deltas, coordinates, target_shapes),
            "adapter": "huggingface_causal_lm",
            "autograd": True,
            "real_forward": True,
            "device": str(device),
            "model_dtype": str(dtype).replace("torch.", "") if dtype is not None else "default",
            "initial_loss": initial_loss,
            "perplexity": exp(min(final_loss, 20.0)),
            "validation_loss": validation_loss,
            "validation_perplexity": exp(min(validation_loss, 20.0)),
            "tokens_per_s": tokens_seen / train_elapsed,
            "tokens_seen": tokens_seen,
            "batch_count": len(train_batches),
            "cuda_peak_bytes": cuda_peak,
            "load_cuda_peak_bytes": load_cuda_peak,
            "routing_cuda_peak_bytes": routing_cuda_peak,
            "train_cuda_peak_bytes": train_cuda_peak,
            "delta_payload_format": "saint_sparse_delta",
            "routing_method": routing_method,
            "delta_application": delta_application,
            "routing_max_length": routing_length,
            "routing_batch_size": routing_batch_size,
            "target_matrices": names,
            "marco": "fase_13_marco_3",
        },
    )


__all__ = ["run_hf_forward"]
