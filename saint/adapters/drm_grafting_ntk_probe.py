"""NTK-Mirror-inspired activation gate probes for DRM-G routing."""

from __future__ import annotations

from typing import Any


def temporarily_enable_first_parameter(model):
    for param in model.parameters():
        old = bool(param.requires_grad)
        if not old:
            param.requires_grad_(True)
        return param, old
    return None, None


def activation_gate_scores_for_loss(
    torch,
    model,
    target_names: list[str],
    loss_fn,
    *,
    stage: int | None = None,
    batch_index: int | None = None,
) -> list[dict[str, Any]]:
    """Score target modules by NTK-Mirror-style abs(grad_h * h)."""
    modules = dict(model.named_modules())
    missing = [name for name in target_names if name not in modules]
    if missing:
        raise ValueError(f"unknown ntk activation probe target(s): {missing}")
    captured: dict[str, Any] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if getattr(h, "requires_grad", False):
                h.retain_grad()
            captured[name] = h
            return output
        return hook

    for name in target_names:
        handles.append(modules[name].register_forward_hook(make_hook(name)))
    enabled_param, old_requires_grad = temporarily_enable_first_parameter(model)
    try:
        model.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        rows = []
        for name in target_names:
            rows.append(_score_captured_activation(torch, name, captured, stage, batch_index))
        rows.sort(key=lambda row: row["ntk_activation_score"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["ntk_rank"] = rank
        return rows
    finally:
        for handle in handles:
            handle.remove()
        if enabled_param is not None:
            enabled_param.requires_grad_(old_requires_grad)
        model.zero_grad(set_to_none=True)


def _score_captured_activation(torch, name: str, captured: dict[str, Any], stage, batch_index):
    h = captured.get(name)
    grad = getattr(h, "grad", None) if h is not None else None
    if h is None or grad is None:
        total = 0.0
        channel_scores = torch.zeros((0,), dtype=torch.float32)
    else:
        contribution = (grad.detach().float() * h.detach().float()).abs()
        if contribution.ndim == 0:
            channel_scores = contribution.reshape(1)
        elif contribution.ndim == 1:
            channel_scores = contribution
        else:
            channel_scores = contribution.reshape(-1, contribution.shape[-1]).sum(dim=0)
        total = float(channel_scores.sum().item())
    top_channel = None
    top_channel_score = 0.0
    if int(channel_scores.numel()) > 0:
        top_index = int(torch.argmax(channel_scores).item())
        top_channel = top_index
        top_channel_score = float(channel_scores[top_index].item())
    row = {
        "target": name,
        "ntk_activation_score": total,
        "channel_count": int(channel_scores.numel()),
        "top_channel": top_channel,
        "top_channel_score": top_channel_score,
    }
    if stage is not None:
        row["stage"] = int(stage)
    if batch_index is not None:
        row["batch_index"] = int(batch_index)
    return row


def aggregate_ntk_rows(rows: list[dict[str, Any]], stage: int, split: str) -> list[dict[str, Any]]:
    by_target: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = row["target"]
        slot = by_target.setdefault(target, {
            "stage": stage,
            "target": target,
            "ntk_activation_score": 0.0,
            "probe_batches": 0,
            "channel_count": row["channel_count"],
            "top_channel": row["top_channel"],
            "top_channel_score": row["top_channel_score"],
            "split": split,
        })
        slot["ntk_activation_score"] += float(row["ntk_activation_score"])
        slot["probe_batches"] += 1
        if float(row["top_channel_score"]) > float(slot["top_channel_score"]):
            slot["top_channel"] = row["top_channel"]
            slot["top_channel_score"] = row["top_channel_score"]
    ranked = list(by_target.values())
    for row in ranked:
        row["mean_ntk_activation_score"] = row["ntk_activation_score"] / max(1, int(row["probe_batches"]))
    ranked.sort(key=lambda row: row["mean_ntk_activation_score"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["ntk_rank"] = rank
    return ranked


def run_activation_probe_stage(
    torch,
    model,
    grafts,
    metadata: dict[str, Any],
    drm_config,
    args,
    accepted: set[int],
    accepted_target_map: dict[int, str],
    targets: list[str],
    stage: int,
    *,
    copy_indices,
    set_state,
    attach_target_map,
    batch_fn,
    tokens_fn,
    loss_fn,
) -> list[dict[str, Any]]:
    batches = int(getattr(args, "ntk_activation_probe_batches", 0) or 0)
    if batches <= 0:
        return []
    split = str(getattr(args, "ntk_activation_probe_split", "train") or "train")
    set_state(grafts, accepted, set())
    handles = attach_target_map(model, grafts, accepted_target_map) if accepted_target_map else []
    rows: list[dict[str, Any]] = []
    try:
        for batch_index in range(batches):
            local = batch_fn(metadata, batch_index)
            local["validation_split"] = split
            seed_key = "data_seed"
            if split != "train":
                local[f"{split}_token_offset"] = batch_index * 4096
                seed_key = "validation_seed"
            inputs, labels = tokens_fn(
                torch,
                local,
                drm_config.vocab_size,
                str(metadata["device"]),
                seed_key=seed_key,
            )
            rows.extend(activation_gate_scores_for_loss(
                torch,
                model,
                targets,
                lambda inputs=inputs, labels=labels: loss_fn(model, inputs, labels),
                stage=stage,
                batch_index=batch_index,
            ))
    finally:
        for handle in handles:
            handle.remove()
    return aggregate_ntk_rows(rows, stage, split) if rows else []


__all__ = [
    "activation_gate_scores_for_loss",
    "aggregate_ntk_rows",
    "run_activation_probe_stage",
]
