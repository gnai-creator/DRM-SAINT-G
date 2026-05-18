"""DRM-G grafting experiments for the DRM 3.5M baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from saint.config import RuntimeConfig
from saint.adapters.drm_grafting_data import token_batch
from saint.adapters.drm_grafting_decision import (
    consolidation_payload,
    evaluate_graft_decision,
)
from saint.adapters.drm_grafting_modules import DenseBudgetGraft, PhiHiddenGraft
from saint.adapters.drm_grafting_optimizer import optimizer_to_payload
from saint.transformer.training import MiniTransformerResult


BASELINE_CONFIG = Path("configs/baselines/small_3.5M.yaml")
CONFIG_KEYS = {
    "vocab_size",
    "max_seq_len",
    "d_model",
    "n_layers",
    "n_heads",
    "d_ff",
    "dropout",
    "bias",
    "d_manifold",
    "metric_hidden",
    "metric_rank",
    "n_anchors",
    "gamma_enabled",
    "gamma_c",
    "gamma_alpha",
    "gravity_enabled",
    "gravity_strength",
    "gravity_n_rff",
    "variable_dim",
}


@dataclass(frozen=True)
class DRMGraftConfig:
    baseline_config: str
    phi_rank: int
    target_module: str
    scale: float
    trainable_parameters: int


def _default_drm_root() -> Path:
    return Path(__file__).resolve().parents[3] / "drm_transformer"


def _coerce_scalar(value: str) -> Any:
    raw = value.split("#", 1)[0].strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.lower() in {"none", "null"}:
        return None
    normalized = raw.replace("_", "")
    try:
        return int(normalized)
    except ValueError:
        try:
            return float(normalized)
        except ValueError:
            return raw.strip("'\"")


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        values[key.strip()] = _coerce_scalar(value)
    return values


def _import_drm(metadata: dict[str, Any]):
    drm_root = Path(metadata.get("drm_root") or _default_drm_root()).resolve()
    drm_src = Path(metadata.get("drm_src") or (drm_root / "src")).resolve()
    if str(drm_src) not in sys.path:
        sys.path.insert(0, str(drm_src))
    try:
        import torch
        from drm_transformer.config import DRMTransformerConfig
        from drm_transformer.model import DRMTransformer
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and drm_transformer are required for DRM-G grafting."
        ) from exc
    return torch, DRMTransformerConfig, DRMTransformer, drm_root


def _baseline_path(metadata: dict[str, Any], drm_root: Path) -> Path:
    configured = metadata.get("baseline_config")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else (drm_root / path)
    return drm_root / BASELINE_CONFIG


def load_drm_baseline_config(metadata: dict[str, Any], config_cls) -> Any:
    _torch, _config_cls, _model_cls, drm_root = _import_drm(metadata)
    data = _read_simple_yaml(_baseline_path(metadata, drm_root))
    overrides = metadata.get("config_overrides", {})
    if isinstance(overrides, dict):
        data.update(overrides)
    fields = {key: data[key] for key in CONFIG_KEYS if key in data}
    return config_cls(**fields)


def _freeze(model) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _tokens(torch, metadata: dict[str, Any], vocab_size: int, device: str, *, seed_key: str = "data_seed"):
    return token_batch(torch, metadata, vocab_size, device, seed_key=seed_key)


def _loss(model, inputs, targets) -> Any:
    _logits, loss = model(inputs, targets)
    if loss is None:
        raise ValueError("DRM model did not return loss")
    return loss


def _train(
    model,
    graft,
    torch,
    inputs,
    targets,
    eval_inputs,
    eval_targets,
    steps: int,
    lr: float,
) -> tuple[float, float, dict[str, Any]]:
    optimizer = torch.optim.AdamW(graft.parameters(), lr=lr)
    for _ in range(max(1, steps)):
        optimizer.zero_grad()
        loss = _loss(model, inputs, targets)
        loss.backward()
        optimizer.step()
    train_loss = float(_loss(model, inputs, targets).detach().cpu().item())
    eval_loss = float(_loss(model, eval_inputs, eval_targets).detach().cpu().item())
    return train_loss, eval_loss, optimizer_to_payload(optimizer)


def _target_module(model, target_module: str):
    modules = dict(model.named_modules())
    if target_module not in modules:
        available = sorted(name for name in modules if name)
        raise ValueError(f"unknown graft target_module: {target_module}; available={available[:12]}")
    return modules[target_module]


def _capture_signal(torch, model, target_module: str, inputs, targets) -> tuple[Any, Any]:
    captured: dict[str, Any] = {}

    def hook(_module, _hook_inputs, output):
        if not hasattr(output, "retain_grad"):
            raise TypeError("graft target output must be a tensor")
        output.retain_grad()
        captured["activation"] = output
        return output

    handle = _target_module(model, target_module).register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        loss = _loss(model, inputs, targets)
        loss.backward()
        activation = captured["activation"].detach()
        gradient = captured["activation"].grad.detach()
        return activation.reshape(-1, activation.shape[-1]), gradient.reshape(-1, gradient.shape[-1])
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def _orthogonal_basis(torch, signal, rank: int) -> Any:
    signal = signal.detach().float().cpu()
    signal = signal - signal.mean(dim=0, keepdim=True)
    if not torch.isfinite(signal).all() or float(signal.abs().sum()) == 0.0:
        return None
    try:
        _u, _s, vh = torch.linalg.svd(signal, full_matrices=False)
        basis = vh[:rank].transpose(0, 1).contiguous()
    except RuntimeError:
        scores = signal.abs().sum(dim=0)
        indices = torch.topk(scores, k=min(rank, signal.shape[1])).indices
        basis = torch.zeros(signal.shape[1], rank)
        for col, index in enumerate(indices):
            basis[int(index), col] = 1.0
    if basis.shape[1] < rank:
        pad = torch.zeros(basis.shape[0], rank - basis.shape[1])
        basis = torch.cat([basis, pad], dim=1)
    return basis[:, :rank]


def _projection_init(
    torch,
    model_cls,
    drm_config,
    metadata: dict[str, Any],
    device: str,
    train_inputs,
    train_targets,
    rank: int,
    seed: int,
) -> tuple[Any, Any] | None:
    mode = str(metadata.get("projection_init", "random"))
    if mode == "random":
        return None
    torch.manual_seed(seed)
    model = model_cls(drm_config).to(device)
    model.train()
    target_module = str(metadata.get("target_module", "final_norm"))
    activation, gradient = _capture_signal(torch, model, target_module, train_inputs, train_targets)
    if mode == "activation":
        signal = activation
    elif mode == "gradient":
        signal = gradient
    elif mode in {"activation_gradient", "actgrad"}:
        signal = activation * gradient.abs()
    else:
        raise ValueError(f"unknown projection_init: {mode}")
    basis = _orthogonal_basis(torch, signal, rank)
    if basis is None:
        return None
    return basis, basis.transpose(0, 1).contiguous()


def _run_one(
    torch,
    model_cls,
    drm_config,
    graft,
    device: str,
    inputs,
    targets,
    eval_inputs,
    eval_targets,
    steps: int,
    lr: float,
    seed: int,
    target_module: str,
) -> tuple[float, float, dict[str, Any]]:
    torch.manual_seed(seed)
    model = model_cls(drm_config).to(device)
    model.eval()
    _freeze(model)
    graft.to(device)
    handle = _target_module(model, target_module).register_forward_hook(graft.hook)
    try:
        initial = float(_loss(model, eval_inputs, eval_targets).detach().cpu().item())
        _train_final, final, optimizer_payload = _train(
            model,
            graft,
            torch,
            inputs,
            targets,
            eval_inputs,
            eval_targets,
            steps,
            lr,
        )
    finally:
        handle.remove()
    return initial, final, optimizer_payload


def _load_optional_state(model, metadata: dict[str, Any], torch) -> None:
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


def _consolidation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "delta_weight"}


def inspect_graft_model(config: RuntimeConfig) -> dict:
    metadata = dict(config.metadata or {})
    torch, config_cls, model_cls, drm_root = _import_drm(metadata)
    drm_config = load_drm_baseline_config(metadata, config_cls)
    model = model_cls(drm_config)
    total = sum(param.numel() for param in model.parameters())
    rank = int(metadata.get("phi_rank", 8))
    return {
        "task": config.task,
        "adapter": "drm_grafting",
        "baseline_config": str(_baseline_path(metadata, drm_root)),
        "d_model": drm_config.d_model,
        "n_layers": drm_config.n_layers,
        "vocab_size": drm_config.vocab_size,
        "base_parameters": int(total),
        "graft_parameters": rank * rank,
        "torch_available": torch is not None,
    }


def run_drm_graft(config: RuntimeConfig) -> MiniTransformerResult:
    start = perf_counter()
    metadata = dict(config.metadata or {})
    torch, config_cls, model_cls, drm_root = _import_drm(metadata)
    seed = int(metadata.get("seed", config.seed))
    torch.manual_seed(seed)
    device = str(metadata.get("device", "cpu"))
    drm_config = load_drm_baseline_config(metadata, config_cls)
    rank = int(metadata.get("phi_rank", 8))
    scale = float(metadata.get("graft_scale", 1.0))
    lr = float(metadata.get("learning_rate", 0.05))
    inputs, targets = _tokens(torch, metadata, drm_config.vocab_size, device)
    eval_inputs, eval_targets = _tokens(
        torch,
        metadata,
        drm_config.vocab_size,
        device,
        seed_key="validation_seed",
    )

    target_module = str(metadata.get("target_module", "final_norm"))
    projections = _projection_init(
        torch, model_cls, drm_config, metadata, device, inputs, targets, rank, seed
    )
    phi = PhiHiddenGraft(torch, drm_config.d_model, rank, scale, seed, projections)
    dense = DenseBudgetGraft(torch, drm_config.d_model, rank, scale)
    torch.manual_seed(seed)
    base_model = model_cls(drm_config).to(device)
    _load_optional_state(base_model, metadata, torch)
    base_model.eval()
    base_loss = float(_loss(base_model, eval_inputs, eval_targets).detach().cpu().item())
    del base_model

    initial_loss, graft_loss, optimizer_payload = _run_one(
        torch,
        model_cls,
        drm_config,
        phi,
        device,
        inputs,
        targets,
        eval_inputs,
        eval_targets,
        config.steps,
        lr,
        seed,
        target_module,
    )
    _dense_initial, dense_loss, _dense_optimizer_payload = _run_one(
        torch,
        model_cls,
        drm_config,
        dense,
        device,
        inputs,
        targets,
        eval_inputs,
        eval_targets,
        config.steps,
        lr,
        seed,
        target_module,
    )
    parameter_count = rank * rank
    validation_gain = base_loss - graft_loss
    dense_gain = base_loss - dense_loss
    projection_mode = str(metadata.get("projection_init", "random"))
    graft_payload = phi.payload(target_module, projection_mode)
    decision_input = {
        **metadata,
        "validation_gain": validation_gain,
        "validation_gain_per_parameter": validation_gain / max(1, parameter_count),
        "dense_budget_gain": dense_gain,
    }
    graft_decision = evaluate_graft_decision(decision_input)
    consolidate_model = model_cls(drm_config)
    consolidation = consolidation_payload(torch, consolidate_model, graft_payload)
    marco = str(
        metadata.get(
            "marco",
            "drm_g_marco_1" if projection_mode == "random" else "drm_g_marco_2",
        )
    )
    return MiniTransformerResult(
        name="drm_g_saint_phi_graft",
        train_loss=graft_loss,
        test_loss=graft_loss,
        parameter_count=parameter_count,
        optimizer_state_values=parameter_count * 2,
        elapsed_s=perf_counter() - start,
        metadata={
            "baseline_config": str(_baseline_path(metadata, drm_root)),
            "base_parameters": inspect_graft_model(config)["base_parameters"],
            "base_loss": base_loss,
            "initial_graft_loss": initial_loss,
            "dense_budget_loss": dense_loss,
            "validation_gain": validation_gain,
            "validation_gain_per_parameter": validation_gain / max(1, parameter_count),
            "dense_budget_gain": dense_gain,
            "dense_budget_gain_per_parameter": dense_gain / max(1, parameter_count),
            "delta_payload": graft_payload,
            "optimizer_state_payload": optimizer_payload,
            "graft_decision": graft_decision,
            "graft_approved": graft_decision["approved"],
            "graft_decision_status": graft_decision["decision"],
            "consolidation": _consolidation_summary(consolidation),
            "consolidation_payload": consolidation,
            "phi_rank": rank,
            "projection_init": projection_mode,
            "graft_scale": scale,
            "target_module": target_module,
            "frozen_base": True,
            "drm_g": True,
            "marco": marco,
            "config_summary": {
                "vocab_size": drm_config.vocab_size,
                "d_model": drm_config.d_model,
                "n_layers": drm_config.n_layers,
                "n_heads": drm_config.n_heads,
                "d_ff": drm_config.d_ff,
            },
        },
    )


__all__ = [
    "DRMGraftConfig",
    "inspect_graft_model",
    "run_drm_graft",
]
