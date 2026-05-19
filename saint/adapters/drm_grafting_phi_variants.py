"""A Phi B graft variants for DRM-G benchmarks."""

from __future__ import annotations

from typing import Any


def capture_activation_gradient(torch, model, target_module, loss_fn, inputs, targets):
    captured: dict[str, Any] = {}

    def hook(_module, _hook_inputs, output):
        output.retain_grad()
        captured["activation"] = output
        return output

    handle = target_module.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, inputs, targets)
        loss.backward()
        activation = captured["activation"].detach()
        gradient = captured["activation"].grad.detach()
        return activation.reshape(-1, activation.shape[-1]), gradient.reshape(-1, gradient.shape[-1])
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def orthogonal_basis(torch, signal, rank: int):
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


def least_squares_phi(torch, activation, gradient, left, right, step_scale: float):
    left = left.to(activation.device)
    right = right.to(activation.device)
    projected = activation.float().matmul(left.float())
    target = -float(step_scale) * gradient.float()
    try:
        phi = torch.linalg.pinv(projected).matmul(target).matmul(torch.linalg.pinv(right.float()))
    except RuntimeError:
        phi = torch.zeros(left.shape[1], right.shape[0])
    if not torch.isfinite(phi).all():
        phi = torch.zeros(left.shape[1], right.shape[0])
    return phi


class PhiVariantGraft:
    def __init__(
        self,
        torch,
        left,
        right,
        phi,
        *,
        scale: float = 1.0,
        train_ab: bool = False,
        residual_indices=None,
    ):
        self.torch = torch
        parameter = torch.nn.Parameter
        self.left = parameter(left.clone()) if train_ab else left.clone()
        self.right = parameter(right.clone()) if train_ab else right.clone()
        self.phi = parameter(phi.clone())
        self.scale = float(scale)
        self.train_ab = train_ab
        if residual_indices is None:
            self.residual_indices = None
            self.residual_values = None
        else:
            indices = torch.tensor(residual_indices, dtype=torch.long)
            self.residual_indices = indices
            self.residual_values = parameter(torch.zeros(len(residual_indices)))

    def to(self, device: str):
        if self.train_ab:
            self.left = self.torch.nn.Parameter(self.left.detach().to(device))
            self.right = self.torch.nn.Parameter(self.right.detach().to(device))
        else:
            self.left = self.left.to(device)
            self.right = self.right.to(device)
        self.phi = self.torch.nn.Parameter(self.phi.detach().to(device))
        if self.residual_indices is not None:
            self.residual_indices = self.residual_indices.to(device)
            self.residual_values = self.torch.nn.Parameter(self.residual_values.detach().to(device))
        return self

    def parameters(self):
        params = [self.phi]
        if self.train_ab:
            params.extend([self.left, self.right])
        if self.residual_values is not None:
            params.append(self.residual_values)
        return params

    def parameter_count(self) -> int:
        return sum(int(param.numel()) for param in self.parameters())

    def hook(self, _module, _inputs, output):
        delta = output.matmul(self.left).matmul(self.phi).matmul(self.right)
        if self.residual_indices is not None and len(self.residual_indices) > 0:
            residual = self.torch.zeros(output.shape[-1], device=output.device, dtype=output.dtype)
            residual[self.residual_indices] = self.residual_values.to(output.dtype)
            delta = delta + output * residual
        return output + self.scale * delta


def make_phi_variant(
    torch,
    activation,
    gradient,
    rank: int,
    *,
    init: str,
    train_ab: bool = False,
    residual_k: int = 0,
    step_scale: float = 1.0,
):
    basis = orthogonal_basis(torch, gradient, rank)
    if basis is None:
        basis = torch.eye(gradient.shape[-1], rank)
    left = basis.float()
    right = basis.transpose(0, 1).contiguous().float()
    if init == "least_squares_gradient":
        phi = least_squares_phi(torch, activation, gradient, left, right, step_scale)
    elif init == "gradient_diag":
        left = left.to(activation.device)
        scores = (activation.float() * gradient.float()).mean(dim=0)
        phi = torch.diag(left.transpose(0, 1).matmul(-scores))
    elif init == "zero":
        phi = torch.zeros(rank, rank)
    else:
        raise ValueError(f"unknown phi init: {init}")
    residual_indices = None
    if residual_k > 0:
        scores = gradient.float().abs().mean(dim=0)
        residual_indices = torch.topk(scores, k=min(residual_k, scores.numel())).indices.tolist()
    return PhiVariantGraft(
        torch,
        left,
        right,
        phi,
        train_ab=train_ab,
        residual_indices=residual_indices,
    )


__all__ = [
    "PhiVariantGraft",
    "capture_activation_gradient",
    "make_phi_variant",
]
