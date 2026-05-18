"""PyTorch autograd smoke path for the drm_transformer adapter."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from saint.config import RuntimeConfig
from saint.transformer.training import MiniTransformerResult


DEFAULT_KEYWORDS = (
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
    "attn.out_proj.weight",
    "ffn.up_proj.weight",
    "ffn.down_proj.weight",
    "dim_gate.gate_net.0.weight",
)


def _keywords(metadata: dict[str, Any]) -> tuple[str, ...]:
    values = metadata.get("keywords", DEFAULT_KEYWORDS)
    if isinstance(values, list):
        return tuple(str(item) for item in values)
    return DEFAULT_KEYWORDS


def _matches(name: str, keywords: tuple[str, ...]) -> bool:
    return not keywords or any(keyword in name for keyword in keywords)


def _import_drm(metadata: dict[str, Any]):
    drm_src = metadata.get("drm_src")
    if drm_src:
        src_path = str(Path(drm_src).resolve())
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
    try:
        import torch
        from drm_transformer.config import DRMTransformerConfig
        from drm_transformer.model import DRMTransformer
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and drm_transformer are required for drm_saint_autograd_smoke."
        ) from exc
    return torch, DRMTransformerConfig, DRMTransformer


def _tiny_config(metadata: dict[str, Any], config_cls):
    fields = {
        "vocab_size": int(metadata.get("vocab_size", 64)),
        "max_seq_len": int(metadata.get("seq_len", 8)),
        "d_model": int(metadata.get("d_model", 16)),
        "n_layers": int(metadata.get("n_layers", 1)),
        "n_heads": int(metadata.get("n_heads", 2)),
        "d_ff": int(metadata.get("d_ff", 32)),
        "dropout": float(metadata.get("dropout", 0.0)),
        "d_manifold": int(metadata.get("d_manifold", 4)),
        "metric_hidden": int(metadata.get("metric_hidden", 8)),
        "metric_rank": int(metadata.get("metric_rank", 2)),
        "n_anchors": int(metadata.get("n_anchors", 2)),
        "gravity_enabled": bool(metadata.get("gravity_enabled", False)),
        "gamma_enabled": bool(metadata.get("gamma_enabled", False)),
        "variable_dim": bool(metadata.get("variable_dim", False)),
    }
    return config_cls(**fields)


def generated_matrix_payload(config: RuntimeConfig) -> dict[str, list[list[float]]]:
    metadata = dict(config.metadata or {})
    torch, config_cls, model_cls = _import_drm(metadata)
    torch.manual_seed(int(metadata.get("seed", config.seed)))
    model = model_cls(_tiny_config(metadata, config_cls))
    keywords = _keywords(metadata)
    matrices = {
        name: param.detach().cpu().float().tolist()
        for name, param in model.named_parameters()
        if param.ndim == 2 and _matches(name, keywords)
    }
    if not matrices:
        raise ValueError("no generated DRM matrices matched SAINT keywords")
    return matrices


def _make_token_batch(torch, metadata: dict[str, Any], vocab_size: int, device: str):
    batch_size = int(metadata.get("batch_size", 2))
    seq_len = int(metadata.get("seq_len", 8))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(metadata.get("data_seed", 123)))
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=generator)
    return tokens[:, :-1].contiguous().to(device), tokens[:, 1:].contiguous().to(device)


def _load_optional_model_state(model, metadata: dict[str, Any], torch) -> None:
    checkpoint = metadata.get("checkpoint")
    if not checkpoint:
        return
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    if isinstance(state, dict):
        model.load_state_dict(state, strict=False)


def _trainable_named_parameters(model, keywords: tuple[str, ...]):
    return [
        (name, param)
        for name, param in model.named_parameters()
        if param.ndim == 2 and _matches(name, keywords)
    ]


def _block_regions(named_params, block_size: int) -> list[dict]:
    regions = []
    for name, param in named_params:
        if param.grad is None:
            continue
        rows, cols = int(param.shape[0]), int(param.shape[1])
        grad = param.grad.detach()
        for row in range(0, rows, block_size):
            for col in range(0, cols, block_size):
                row_end = min(row + block_size, rows)
                col_end = min(col + block_size, cols)
                regions.append(
                    {
                        "matrix": name,
                        "row": row,
                        "col": col,
                        "rows": row_end - row,
                        "cols": col_end - col,
                        "sensitivity": grad[row:row_end, col:col_end].abs().sum().item(),
                    }
                )
    return sorted(regions, key=lambda item: item["sensitivity"], reverse=True)


def _select_regions(regions: list[dict], parameter_budget: int) -> list[dict]:
    selected = []
    used = 0
    for region in regions:
        cost = region["rows"] * region["cols"]
        if used + cost > parameter_budget:
            continue
        selected.append(region)
        used += cost
        if used >= parameter_budget:
            break
    return selected


def _mask_map(torch, named_params, selected: list[dict]) -> dict[str, Any]:
    masks = {name: torch.zeros_like(param) for name, param in named_params}
    for region in selected:
        mask = masks[region["matrix"]]
        row = region["row"]
        col = region["col"]
        mask[row:row + region["rows"], col:col + region["cols"]] = 1.0
    return masks


def _snapshot(named_params) -> dict[str, Any]:
    return {name: param.detach().clone() for name, param in named_params}


def _delta_payload(named_params, initial: dict[str, Any]) -> dict[str, list[list[float]]]:
    return {
        name: (param.detach().cpu() - initial[name].cpu()).tolist()
        for name, param in named_params
    }


def _loss_value(model, inputs, targets) -> Any:
    _logits, loss = model(inputs, targets)
    if loss is None:
        raise ValueError("drm_transformer did not return a loss")
    return loss


def _train_full_baseline(model, torch, inputs, targets, steps: int, learning_rate: float) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = _loss_value(model, inputs, targets)
        loss.backward()
        optimizer.step()
    return float(_loss_value(model, inputs, targets).detach().cpu().item())


def run_drm_autograd(config: RuntimeConfig) -> MiniTransformerResult:
    start = perf_counter()
    metadata = dict(config.metadata or {})
    torch, config_cls, model_cls = _import_drm(metadata)
    device = str(metadata.get("device", "cpu"))
    seed = int(metadata.get("seed", config.seed))
    torch.manual_seed(seed)
    drm_config = _tiny_config(metadata, config_cls)
    model = model_cls(drm_config).to(device)
    model.train()
    _load_optional_model_state(model, metadata, torch)
    inputs, targets = _make_token_batch(torch, metadata, drm_config.vocab_size, device)
    named_params = _trainable_named_parameters(model, _keywords(metadata))
    if not named_params:
        raise ValueError("no trainable DRM matrices matched SAINT keywords")

    initial_params = _snapshot(named_params)
    initial_loss = _loss_value(model, inputs, targets)
    initial_loss.backward()
    block_size = int(metadata.get("block_size", 2))
    regions = _block_regions(named_params, block_size)
    selected = _select_regions(regions, max(1, config.parameter_budget))
    masks = _mask_map(torch, named_params, selected)
    model.zero_grad(set_to_none=True)

    learning_rate = float(metadata.get("learning_rate", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(max(1, config.steps)):
        optimizer.zero_grad()
        loss = _loss_value(model, inputs, targets)
        loss.backward()
        for name, param in named_params:
            if param.grad is not None:
                param.grad.mul_(masks[name])
        optimizer.step()

    final_loss = float(_loss_value(model, inputs, targets).detach().cpu().item())
    torch.manual_seed(seed)
    baseline = model_cls(drm_config).to(device)
    baseline.train()
    _load_optional_model_state(baseline, metadata, torch)
    baseline_loss = _train_full_baseline(
        baseline, torch, inputs, targets, max(1, config.steps), learning_rate
    )
    parameter_count = sum(region["rows"] * region["cols"] for region in selected)
    return MiniTransformerResult(
        name="drm_saint_autograd_smoke",
        train_loss=final_loss,
        test_loss=final_loss,
        parameter_count=parameter_count,
        optimizer_state_values=parameter_count * 2,
        elapsed_s=perf_counter() - start,
        metadata={
            "delta_payload": _delta_payload(named_params, initial_params),
            "initial_loss": float(initial_loss.detach().cpu().item()),
            "full_baseline_loss": baseline_loss,
            "selected_regions": selected,
            "available_regions": len(regions),
            "block_size": block_size,
            "autograd": True,
            "sensitivity_method": "gradient_abs_sum",
            "marco": "fase_9_marco_2",
        },
    )


__all__ = ["generated_matrix_payload", "run_drm_autograd"]
