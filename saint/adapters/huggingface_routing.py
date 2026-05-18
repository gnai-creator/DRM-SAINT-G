"""Routing helpers for Hugging Face SAINT forward experiments."""

from __future__ import annotations

from typing import Any, Callable


def _score_indices(torch, scores: dict[str, Any], *, budget: int):
    total = sum(score.numel() for score in scores.values())
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
    elif routing_method == "activation":
        scores = _activation_scores(torch, model, names, input_ids, attention_mask, hybrid=False)
    elif routing_method == "magnitude_activation":
        scores = _activation_scores(torch, model, names, input_ids, attention_mask, hybrid=True)
    elif routing_method == "lm_head_proxy":
        scores = _lm_head_proxy_scores(
            torch, functional_call, model, names, input_ids, attention_mask, loss_fn
        )
    else:
        scores = {name: named[name].detach().abs().cpu() for name in names}
    selected = _score_indices(torch, scores, budget=parameter_budget)
    deltas = {}
    coordinates = {}
    for name in names:
        rows, cols = selected.get(name, (None, None))
        if rows is None:
            continue
        deltas[name] = torch.zeros(rows.numel(), device=named[name].device, requires_grad=True)
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
