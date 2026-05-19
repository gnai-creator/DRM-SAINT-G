"""Residual graft blocks for Phase 16 scale-up experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GraftBlockPlan:
    base_parameters: int
    target_total_parameters: int
    graft_count: int
    d_model: int
    hidden_size: int
    parameters_per_graft: int
    graft_parameters: int
    effective_total_parameters: int
    remaining_gap: int


def plan_graft_blocks(
    *,
    base_parameters: int,
    target_total_parameters: int,
    d_model: int,
    graft_count: int,
) -> GraftBlockPlan:
    """Plan equal-size residual graft blocks for a target parameter count."""
    if graft_count < 1:
        raise ValueError("graft_count must be >= 1")
    if d_model < 1:
        raise ValueError("d_model must be >= 1")
    gap = max(0, int(target_total_parameters) - int(base_parameters))
    per_graft_budget = max(1, gap // int(graft_count))
    hidden_size = max(1, per_graft_budget // (2 * int(d_model)))
    parameters_per_graft = (2 * int(d_model) * hidden_size) + 1
    graft_parameters = parameters_per_graft * int(graft_count)
    effective_total = int(base_parameters) + graft_parameters
    return GraftBlockPlan(
        base_parameters=int(base_parameters),
        target_total_parameters=int(target_total_parameters),
        graft_count=int(graft_count),
        d_model=int(d_model),
        hidden_size=int(hidden_size),
        parameters_per_graft=int(parameters_per_graft),
        graft_parameters=int(graft_parameters),
        effective_total_parameters=int(effective_total),
        remaining_gap=int(target_total_parameters) - int(effective_total),
    )


class DRMGraftBlock:
    """Trainable residual adapter: output + scale * down(act(up(output)))."""

    def __init__(
        self,
        torch,
        *,
        d_model: int,
        hidden_size: int,
        seed: int,
        init_scale: float = 0.01,
        activation: str = "silu",
    ):
        self.torch = torch
        self.d_model = int(d_model)
        self.hidden_size = int(hidden_size)
        self.activation = str(activation)
        self.enabled = True
        self.runtime_scale = 1.0
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.up = torch.nn.Parameter(
            torch.randn(self.d_model, self.hidden_size, generator=generator)
            / max(1, self.d_model)
        )
        self.down = torch.nn.Parameter(
            torch.zeros(self.hidden_size, self.d_model)
        )
        self.scale = torch.nn.Parameter(torch.tensor(float(init_scale)))

    def parameters(self):
        return (self.up, self.down, self.scale)

    def to(self, device: str):
        self.up.data = self.up.data.to(device)
        self.down.data = self.down.data.to(device)
        self.scale.data = self.scale.data.to(device)
        return self

    def parameter_count(self) -> int:
        return int(self.up.numel() + self.down.numel() + self.scale.numel())

    def state_dict(self) -> dict[str, Any]:
        return {
            "up": self.up.detach().cpu(),
            "down": self.down.detach().cpu(),
            "scale": self.scale.detach().cpu(),
            "d_model": self.d_model,
            "hidden_size": self.hidden_size,
            "activation": self.activation,
        }

    def load_state_dict(self, state: dict[str, Any], device: str):
        self.up.data = state["up"].to(device)
        self.down.data = state["down"].to(device)
        self.scale.data = state["scale"].to(device)
        self.d_model = int(state.get("d_model", self.up.shape[0]))
        self.hidden_size = int(state.get("hidden_size", self.up.shape[1]))
        self.activation = str(state.get("activation", self.activation))
        return self

    def _activate(self, value):
        if self.activation == "gelu":
            return self.torch.nn.functional.gelu(value)
        if self.activation == "relu":
            return self.torch.relu(value)
        return self.torch.nn.functional.silu(value)

    def hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if not self.enabled:
            return output
        hidden = self._activate(output.matmul(self.up))
        return output + float(self.runtime_scale) * self.scale * hidden.matmul(self.down)


def set_progressive_state(grafts: list[DRMGraftBlock], active_count: int, warmup: int = 1) -> None:
    active_count = max(1, min(int(active_count), len(grafts)))
    warmup = max(1, int(warmup))
    for index, graft in enumerate(grafts):
        graft.enabled = index < active_count
        age = max(0, active_count - index)
        graft.runtime_scale = min(1.0, age / warmup) if graft.enabled else 0.0
        for param in graft.parameters():
            param.requires_grad_(graft.enabled)


def make_graft_blocks(
    torch,
    *,
    d_model: int,
    hidden_size: int,
    graft_count: int,
    seed: int,
    init_scale: float,
    activation: str,
    device: str,
) -> list[DRMGraftBlock]:
    return [
        DRMGraftBlock(
            torch,
            d_model=d_model,
            hidden_size=hidden_size,
            seed=int(seed) + index,
            init_scale=init_scale,
            activation=activation,
        ).to(device)
        for index in range(int(graft_count))
    ]


def attach_graft_blocks(model, target_modules: list[str], grafts: list[DRMGraftBlock]):
    modules = dict(model.named_modules())
    handles = []
    for index, graft in enumerate(grafts):
        target = target_modules[index % len(target_modules)]
        if target not in modules:
            raise ValueError(f"unknown graft target_module: {target}")
        handles.append(modules[target].register_forward_hook(graft.hook))
    return handles


def graft_checkpoint_payload(
    *,
    grafts: list[DRMGraftBlock],
    target_modules: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "drm_saint_g_graftblock_v1",
        "metadata": dict(metadata),
        "target_modules": list(target_modules),
        "grafts": [graft.state_dict() for graft in grafts],
    }
