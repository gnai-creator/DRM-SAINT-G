"""Real Transformers forward path for small Hugging Face SAINT experiments."""

from __future__ import annotations

from math import exp
from time import perf_counter
from typing import Any

from saint.adapters.huggingface_delta_payload import (
    delta_payload,
    delta_structure_metadata,
)
from saint.adapters.huggingface_loading import load_causal_lm, model_dtype
from saint.adapters.huggingface_routing import build_routed_deltas, merged_params
from saint.adapters.huggingface_sparse_train import (
    inplace_loss_value,
    train_inplace_sparse,
)
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


def _cuda_peak(torch, device) -> int:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


def _clear_cuda(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _enable_memory_savers(model, metadata: dict[str, Any]) -> None:
    if bool(metadata.get("gradient_checkpointing", False)) and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False


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
    explicit = metadata.get("target_names")
    params = dict(model.named_parameters())
    if isinstance(explicit, list) and explicit:
        names = [str(name) for name in explicit if str(name) in params]
    else:
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
        names = [
            name
            for name, param in params.items()
            if param.ndim == 2 and any(keyword in name for keyword in keywords)
        ]
    target_device = str(metadata.get("target_device") or "").lower()
    if target_device:
        names = [
            name
            for name in names
            if str(params[name].device).lower().startswith(target_device)
        ]
    return names[: max(1, int(metadata.get("max_trainable_matrices", 2)))]


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


def _target_shapes(model, names: list[str]) -> dict[str, list[int]]:
    params = dict(model.named_parameters())
    return {name: [int(params[name].shape[0]), int(params[name].shape[1])] for name in names}


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
    _enable_memory_savers(model, metadata)
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    load_cuda_peak = _cuda_peak(torch, device)
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
    train_only = bool(metadata.get("train_only", False))
    stage_elapsed: dict[str, float] = {"load_s": perf_counter() - start}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    routing_start = perf_counter()
    route_ids, route_mask = routing_ids, routing_mask
    route_method = routing_method
    if routing_method == "validation_gradient":
        route_ids, route_mask = val_ids, val_mask
        route_method = "gradient_sequential"
    elif routing_method == "validation_magnitude_activation":
        route_ids, route_mask = val_ids, val_mask
        route_method = "magnitude_activation"
    deltas, coordinates = build_routed_deltas(
        torch,
        functional_call,
        model,
        names,
        parameter_budget=max(1, config.parameter_budget),
        input_ids=route_ids,
        attention_mask=route_mask,
        routing_method=route_method,
        loss_fn=_loss,
        validation_batch=(val_ids, val_mask),
        validation_rerank_multiplier=int(
            metadata.get("validation_rerank_multiplier", 4)
        ),
        validation_rerank_chunk_size=int(
            metadata.get("validation_rerank_chunk_size", 256)
        ),
        validation_probe_epsilon=float(
            metadata.get("validation_probe_epsilon", 1e-3)
        ),
        routing_block_size=int(metadata.get("routing_block_size", 1)),
        validation_rerank_max_candidates=metadata.get(
            "validation_rerank_max_candidates"
        ),
        validation_rerank_batch_size=int(
            metadata.get("validation_rerank_batch_size", 1)
        ),
        structured_prototype_count=int(
            metadata.get("structured_prototype_count", 1)
        ),
        structured_prototype_mode=str(
            metadata.get("structured_prototype_mode", "weight_sign")
        ),
        structured_scale_granularity=str(
            metadata.get("structured_scale_granularity", "block")
        ),
        phi_rank=int(metadata.get("phi_rank", 4)),
        phi_variant=str(metadata.get("phi_variant", "dense")),
        phi_source=str(metadata.get("phi_source", "weight")),
    )
    routing_elapsed = perf_counter() - routing_start
    routing_cuda_peak = _cuda_peak(torch, device)
    _check_cuda_budget(torch, device, metadata, "routing")
    _clear_cuda(torch, device)
    learning_rate = float(metadata.get("learning_rate", 1e-3))
    lr_decay = float(metadata.get("lr_decay", 1.0))
    named = dict(model.named_parameters())
    initial_loss = 0.0
    initial_validation_loss = 0.0
    measure_train_only_loss = bool(metadata.get("measure_train_only_loss", False))
    if train_only and measure_train_only_loss:
        with torch.no_grad():
            initial_loss = float(
                _plain_loss(model, input_ids, attention_mask).detach().cpu().item()
            )
            initial_validation_loss = float(
                _plain_loss(model, val_ids, val_mask).detach().cpu().item()
            )
    elif not train_only and delta_application == "inplace":
        initial_loss = inplace_loss_value(
            torch, model, named, deltas, coordinates, (input_ids, attention_mask)
        )
    elif not train_only:
        optimizer = torch.optim.AdamW(list(deltas.values()), lr=learning_rate)
        initial_loss = _loss_value(
            functional_call,
            model,
            merged_params(torch, model, deltas, coordinates),
            input_ids,
            attention_mask,
        )
    elif delta_application != "inplace":
        optimizer = torch.optim.AdamW(list(deltas.values()), lr=learning_rate)
    steps = max(1, int(config.steps))
    train_start = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    final_loss = initial_loss
    model.train()
    if delta_application == "inplace":
        train_info = train_inplace_sparse(
            torch,
            model,
            deltas,
            coordinates,
            train_batches,
            steps=steps,
            learning_rate=learning_rate,
            lr_decay=lr_decay,
            validation_batch=(val_ids, val_mask)
            if bool(metadata.get("validate_during_train", False))
            else None,
            early_stopping=bool(metadata.get("early_stopping", False)),
            min_delta=float(metadata.get("early_stopping_min_delta", 0.0)),
        )
        final_loss = float(train_info["train_loss"])
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
                final_loss = float(loss.detach().cpu().item() * len(train_batches))
                loss.backward()
            optimizer.step()
    model.eval()
    train_elapsed = max(perf_counter() - train_start, 1e-9)
    train_cuda_peak = _cuda_peak(torch, device)
    _check_cuda_budget(torch, device, metadata, "train")
    _clear_cuda(torch, device)
    if train_only and measure_train_only_loss:
        final_loss = inplace_loss_value(
            torch, model, named, deltas, coordinates, (input_ids, attention_mask)
        )
        validation_loss = inplace_loss_value(
            torch, model, named, deltas, coordinates, (val_ids, val_mask)
        )
    elif train_only:
        validation_loss = final_loss
    elif delta_application == "inplace":
        final_loss = inplace_loss_value(
            torch, model, named, deltas, coordinates, (input_ids, attention_mask)
        )
        validation_loss = inplace_loss_value(
            torch, model, named, deltas, coordinates, (val_ids, val_mask)
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
    checkpoint_start = perf_counter()
    payload = delta_payload(deltas, coordinates, target_shapes)
    checkpoint_elapsed = perf_counter() - checkpoint_start
    cuda_peak = _cuda_peak(torch, device)
    stage_elapsed.update(
        {
            "routing_s": routing_elapsed,
            "train_s": train_elapsed,
            "checkpoint_payload_s": checkpoint_elapsed,
        }
    )
    return MiniTransformerResult(
        name="hf_saint_forward_smoke",
        train_loss=final_loss,
        test_loss=final_loss,
        parameter_count=parameter_count,
        optimizer_state_values=parameter_count * 2,
        elapsed_s=perf_counter() - start,
        metadata={
            "delta_payload": payload,
            "delta_structure": delta_structure_metadata(deltas, coordinates),
            "adapter": "huggingface_causal_lm",
            "autograd": True,
            "real_forward": True,
            "device": str(device),
            "model_dtype": str(dtype).replace("torch.", "") if dtype is not None else "default",
            "initial_loss": initial_loss,
            "initial_validation_loss": initial_validation_loss,
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
            "stage_elapsed": stage_elapsed,
            "delta_payload_format": "saint_sparse_delta",
            "routing_method": routing_method,
            "effective_routing_method": route_method,
            "delta_application": delta_application,
            "routing_max_length": routing_length,
            "routing_batch_size": routing_batch_size,
            "routing_block_size": int(metadata.get("routing_block_size", 1)),
            "validation_rerank_batch_size": int(
                metadata.get("validation_rerank_batch_size", 1)
            ),
            "structured_prototype_count": int(
                metadata.get("structured_prototype_count", 1)
            ),
            "structured_prototype_mode": str(
                metadata.get("structured_prototype_mode", "weight_sign")
            ),
            "structured_scale_granularity": str(
                metadata.get("structured_scale_granularity", "block")
            ),
            "phi_rank": int(metadata.get("phi_rank", 4)),
            "phi_variant": str(metadata.get("phi_variant", "dense")),
            "phi_source": str(metadata.get("phi_source", "weight")),
            "target_matrices": names,
            "train_only": train_only,
            "gradient_checkpointing": bool(metadata.get("gradient_checkpointing", False)),
            "lr_decay": lr_decay,
            "validation_history": train_info.get("history", [])
            if delta_application == "inplace"
            else [],
            "steps_ran": train_info.get("steps_ran", steps)
            if delta_application == "inplace"
            else steps,
            "marco": str(metadata.get("marco", "fase_13_marco_3")),
        },
    )


__all__ = ["run_hf_forward"]
