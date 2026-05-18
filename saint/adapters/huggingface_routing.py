"""Routing helpers for Hugging Face SAINT forward experiments."""

from __future__ import annotations

from typing import Any, Callable


def _score_indices(torch, scores: dict[str, Any], *, budget: int):
    total = sum(score.numel() for score in scores.values())
    if total <= 0:
        return {}
    count = max(1, min(budget, total))
    flat = torch.cat([score.detach().abs().cpu().flatten() for score in scores.values()])
    _, selected = torch.topk(flat, k=count)
    offsets = {}
    start = 0
    for name, score in scores.items():
        offsets[name] = (start, start + score.numel(), score.shape)
        start += score.numel()
    indices = {}
    for name, (start, end, shape) in offsets.items():
        local = selected[(selected >= start) & (selected < end)] - start
        if local.numel() > 0:
            indices[name] = torch.unravel_index(local, shape)
    return indices


def _indices_from_flat(torch, selected, offsets):
    indices = {}
    for name, (start, end, shape) in offsets.items():
        local = selected[(selected >= start) & (selected < end)] - start
        if local.numel() > 0:
            indices[name] = torch.unravel_index(local, shape)
    return indices


def _flatten_scores(torch, scores: dict[str, Any], *, budget: int):
    total = sum(score.numel() for score in scores.values())
    if total <= 0:
        return None, {}
    count = max(1, min(budget, total))
    flat = torch.cat([score.detach().abs().cpu().flatten() for score in scores.values()])
    _, selected = torch.topk(flat, k=count)
    offsets = {}
    start = 0
    for name, score in scores.items():
        offsets[name] = (start, start + score.numel(), score.shape)
        start += score.numel()
    return selected, offsets


def _gradient_scores(torch, functional_call, model, names, input_ids, attention_mask, loss_fn):
    params = {
        name: param.detach().requires_grad_(True)
        for name, param in model.named_parameters()
        if name in names
    }
    loss = loss_fn(functional_call, model, params, input_ids, attention_mask)
    grads = torch.autograd.grad(loss, [params[name] for name in names], allow_unused=True)
    return {
        name: grad.detach().abs().cpu() if grad is not None else params[name].detach().abs().cpu()
        for name, grad in zip(names, grads)
    }


def _gradient_scores_sequential(
    torch,
    functional_call,
    model,
    names,
    input_ids,
    attention_mask,
    loss_fn,
):
    named = dict(model.named_parameters())
    scores = {}
    for name in names:
        param = named[name].detach().requires_grad_(True)
        loss = loss_fn(functional_call, model, {name: param}, input_ids, attention_mask)
        grad = torch.autograd.grad(loss, param, allow_unused=True)[0]
        scores[name] = (
            grad.detach().abs().cpu()
            if grad is not None
            else param.detach().abs().cpu()
        )
        del loss, grad, param
        if input_ids.device.type == "cuda":
            torch.cuda.empty_cache()
    return scores


def _module_name(param_name: str) -> str:
    if param_name.endswith(".weight"):
        return param_name[:-7]
    if param_name.endswith(".bias"):
        return param_name[:-5]
    return param_name.rsplit(".", 1)[0]


def _activation_vectors(torch, model, names, input_ids, attention_mask):
    vectors = {}
    handles = []
    module_to_names: dict[str, list[str]] = {}
    for name in names:
        module_to_names.setdefault(_module_name(name), []).append(name)

    def hook(module_name):
        def capture(_module, inputs, _output):
            if not inputs:
                return
            value = inputs[0].detach().abs().float()
            while value.ndim > 1:
                value = value.mean(dim=0)
            for name in module_to_names[module_name]:
                vectors[name] = value.cpu()
        return capture

    for module_name in module_to_names:
        try:
            handles.append(model.get_submodule(module_name).register_forward_hook(hook(module_name)))
        except AttributeError:
            continue
    kwargs = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    with torch.no_grad():
        model(**kwargs)
    for handle in handles:
        handle.remove()
    return vectors


def _broadcast_activation(torch, vector, shape):
    rows, cols = shape
    if vector is None or vector.numel() == 0:
        return torch.ones(rows, cols)
    if vector.numel() == rows:
        return vector[:rows].reshape(rows, 1).repeat(1, cols)
    if vector.numel() == cols:
        return vector[:cols].reshape(1, cols).repeat(rows, 1)
    return torch.ones(rows, cols) * float(vector.mean().item())


def _activation_scores(torch, model, names, input_ids, attention_mask, *, hybrid: bool):
    named = dict(model.named_parameters())
    vectors = _activation_vectors(torch, model, names, input_ids, attention_mask)
    scores = {}
    for name in names:
        base = _broadcast_activation(torch, vectors.get(name), named[name].shape)
        scores[name] = base * named[name].detach().abs().cpu() if hybrid else base
    return scores


def _lm_head_proxy_scores(torch, functional_call, model, names, input_ids, attention_mask, loss_fn):
    lm_names = [name for name in names if "lm_head.weight" in name]
    if not lm_names:
        return _activation_scores(torch, model, names, input_ids, attention_mask, hybrid=True)
    scores = _activation_scores(torch, model, names, input_ids, attention_mask, hybrid=True)
    scores.update(
        _gradient_scores(
            torch,
            functional_call,
            model,
            lm_names,
            input_ids,
            attention_mask,
            loss_fn,
        )
    )
    return scores


def _limit_scores(torch, scores: dict[str, Any], limits: dict[str, tuple[int, int]] | None):
    if not limits:
        return scores
    limited = {}
    for name, score in scores.items():
        rows, cols = limits.get(name, score.shape)
        masked = torch.zeros_like(score)
        masked[: min(rows, score.shape[0]), : min(cols, score.shape[1])] = score[
            : min(rows, score.shape[0]),
            : min(cols, score.shape[1]),
        ]
        limited[name] = masked
    return limited


def _forward_loss_value(torch, model, batch) -> float:
    input_ids, attention_mask = batch
    kwargs = {"input_ids": input_ids, "labels": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    with torch.no_grad():
        return float(model(**kwargs).loss.detach().cpu().item())


def _temporary_coordinate_delta(torch, param, rows, cols, *, epsilon: float, sign: float) -> Any:
    values = torch.sign(param[rows, cols].detach()) * float(sign) * epsilon
    values = values.to(device=param.device, dtype=param.dtype)
    values = torch.where(values == 0, torch.full_like(values, float(sign) * epsilon), values)
    return values


def _validation_rerank_indices(
    torch,
    model,
    names,
    scores: dict[str, Any],
    *,
    budget: int,
    candidate_multiplier: int,
    chunk_size: int,
    validation_batch,
    epsilon: float,
):
    named = dict(model.named_parameters())
    selected, offsets = _flatten_scores(
        torch,
        scores,
        budget=max(budget, budget * max(1, candidate_multiplier)),
    )
    if selected is None:
        return {}
    candidates = _indices_from_flat(torch, selected, offsets)
    baseline = _forward_loss_value(torch, model, validation_batch)
    ranked = []
    for name in names:
        rows, cols = candidates.get(name, (None, None))
        if rows is None:
            continue
        param = named[name]
        rows = rows.to(param.device)
        cols = cols.to(param.device)
        for start in range(0, rows.numel(), max(1, chunk_size)):
            chunk_rows = rows[start:start + max(1, chunk_size)]
            chunk_cols = cols[start:start + max(1, chunk_size)]
            best_gain = None
            for direction in (-1.0, 1.0):
                values = _temporary_coordinate_delta(
                    torch,
                    param,
                    chunk_rows,
                    chunk_cols,
                    epsilon=epsilon,
                    sign=direction,
                )
                with torch.no_grad():
                    param.index_put_((chunk_rows, chunk_cols), values, accumulate=True)
                try:
                    loss = _forward_loss_value(torch, model, validation_batch)
                finally:
                    with torch.no_grad():
                        param.index_put_((chunk_rows, chunk_cols), -values, accumulate=True)
                gain = baseline - loss
                best_gain = gain if best_gain is None else max(best_gain, gain)
                del values
            ranked.append(
                (
                    float(best_gain or 0.0),
                    name,
                    chunk_rows.detach().cpu(),
                    chunk_cols.detach().cpu(),
                )
            )
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen: dict[str, list[Any]] = {}
    remaining = max(1, budget)
    for _gain, name, rows, cols in ranked:
        if remaining <= 0:
            break
        take = min(remaining, rows.numel())
        if take <= 0:
            continue
        chosen.setdefault(name, [[], []])
        chosen[name][0].append(rows[:take])
        chosen[name][1].append(cols[:take])
        remaining -= take
    return {
        name: (torch.cat(parts[0]), torch.cat(parts[1]))
        for name, parts in chosen.items()
        if parts[0]
    }


def build_routed_deltas(
    torch,
    functional_call,
    model,
    names: list[str],
    *,
    parameter_budget: int,
    input_ids,
    attention_mask,
    routing_method: str,
    loss_fn: Callable,
    score_limits: dict[str, tuple[int, int]] | None = None,
    validation_batch=None,
    validation_rerank_multiplier: int = 4,
    validation_rerank_chunk_size: int = 256,
    validation_probe_epsilon: float = 1e-3,
):
    named = dict(model.named_parameters())
    if routing_method == "gradient":
        scores = _gradient_scores(
            torch, functional_call, model, names, input_ids, attention_mask, loss_fn
        )
    elif routing_method == "gradient_sequential":
        scores = _gradient_scores_sequential(
            torch, functional_call, model, names, input_ids, attention_mask, loss_fn
        )
    elif routing_method in {"activation", "activation_validation_rerank"}:
        scores = _activation_scores(torch, model, names, input_ids, attention_mask, hybrid=False)
    elif routing_method == "magnitude_activation":
        scores = _activation_scores(torch, model, names, input_ids, attention_mask, hybrid=True)
    elif routing_method == "lm_head_proxy":
        scores = _lm_head_proxy_scores(
            torch, functional_call, model, names, input_ids, attention_mask, loss_fn
        )
    else:
        scores = {name: named[name].detach().abs().cpu() for name in names}
    if not scores:
        scores = {name: named[name].detach().abs().cpu() for name in names}
    scores = _limit_scores(torch, scores, score_limits)
    if routing_method == "activation_validation_rerank" and validation_batch is not None:
        selected = _validation_rerank_indices(
            torch,
            model,
            names,
            scores,
            budget=parameter_budget,
            candidate_multiplier=validation_rerank_multiplier,
            chunk_size=validation_rerank_chunk_size,
            validation_batch=validation_batch,
            epsilon=validation_probe_epsilon,
        )
    else:
        selected = _score_indices(torch, scores, budget=parameter_budget)
    deltas = {}
    coordinates = {}
    for name in names:
        rows, cols = selected.get(name, (None, None))
        if rows is None:
            continue
        deltas[name] = torch.zeros(
            rows.numel(),
            device=named[name].device,
            dtype=named[name].dtype,
            requires_grad=True,
        )
        coordinates[name] = (rows.to(named[name].device), cols.to(named[name].device))
    return deltas, coordinates


def dense_delta(torch, param, values, coordinates):
    update = torch.zeros_like(param)
    rows, cols = coordinates
    return update.index_put((rows, cols), values)


def merged_params(torch, model, deltas, coordinates):
    params = dict(model.named_parameters())
    return {
        name: params[name] + dense_delta(torch, params[name], delta, coordinates[name])
        for name, delta in deltas.items()
    }


__all__ = ["build_routed_deltas", "merged_params"]
