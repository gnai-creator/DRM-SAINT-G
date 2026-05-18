"""Phi-operator routing for Hugging Face SAINT experiments."""

from __future__ import annotations

from typing import Any

from saint.adapters.huggingface_block_routing import (
    block_candidates,
    block_coords,
    block_validation_indices,
)


def phi_validation_indices(
    torch,
    model,
    names,
    scores: dict[str, Any],
    *,
    budget: int,
    block_size: int,
    phi_rank: int,
    phi_variant: str,
    phi_source: str = "weight",
    candidate_multiplier: int,
    validation_batch,
    epsilon: float,
    max_candidates: int | None,
    batch_size: int = 1,
):
    basis_count = _phi_basis_count(max(1, phi_rank), phi_variant)
    block_budget = max(1, budget // max(1, basis_count))
    max_candidates = max(max_candidates or 0, block_budget)
    selected = block_validation_indices(
        torch,
        model,
        names,
        scores,
        budget=block_budget * block_size * block_size,
        block_size=block_size,
        candidate_multiplier=candidate_multiplier,
        validation_batch=validation_batch,
        epsilon=epsilon,
        max_candidates=max_candidates,
        batch_size=batch_size,
    )
    named = dict(model.named_parameters())
    routed = {}
    for name, (rows, cols) in selected.items():
        param = named[name]
        prototype, scale_ids = _phi_prototypes(
            torch,
            param,
            scores[name],
            rows,
            cols,
            block_size=max(1, block_size),
            rank=max(1, phi_rank),
            variant=phi_variant,
            phi_source=phi_source,
        )
        routed[name] = (rows, cols, prototype, scale_ids)
    return routed


def phi_candidate_summary(
    torch,
    scores: dict[str, Any],
    *,
    budget: int,
    block_size: int,
    phi_rank: int,
    phi_variant: str,
    candidate_multiplier: int,
    max_candidates: int | None,
) -> list[dict[str, Any]]:
    basis_count = _phi_basis_count(max(1, phi_rank), phi_variant)
    block_budget = max(1, budget // max(1, basis_count))
    count = block_budget * max(1, candidate_multiplier)
    if max_candidates is not None:
        count = min(count, max(1, max_candidates))
    return [
        {"score": score, "name": name, "row": row, "col": col}
        for score, name, row, col in block_candidates(
            scores,
            block_size=max(1, block_size),
            count=count,
        )
    ]


def _phi_prototypes(
    torch,
    param,
    score,
    rows,
    cols,
    *,
    block_size: int,
    rank: int,
    variant: str,
    phi_source: str,
):
    grouped = _group_blocks(rows, cols, block_size)
    prototypes = []
    scale_ids = []
    scale_offset = 0
    for row, col in grouped:
        block_rows, block_cols = block_coords(torch, param.shape, row, col, block_size)
        block_rows = block_rows.to(param.device)
        block_cols = block_cols.to(param.device)
        block = _source_block(torch, param, score, block_rows, block_cols, block_size, phi_source)
        left, right = _svd_bases(torch, block, rank=rank)
        basis = _phi_basis(torch, rank, variant, device=block.device, dtype=block.dtype)
        local_proto = _projected_basis(torch, left, basis, right)
        local_rows = []
        local_cols = []
        for local_row in range(block.shape[0]):
            for local_col in range(block.shape[1]):
                local_rows.append(row + local_row)
                local_cols.append(col + local_col)
                prototypes.append(local_proto[:, local_row, local_col])
                scale_ids.append(torch.arange(basis.shape[0]) + scale_offset)
        scale_offset += basis.shape[0]
    prototype = torch.stack(prototypes, dim=0).detach().cpu()
    ids = torch.stack(scale_ids, dim=0).detach().cpu()
    return prototype, ids


def _group_blocks(rows, cols, block_size: int) -> list[tuple[int, int]]:
    seen = []
    values = set()
    for row, col in zip(rows.detach().cpu().tolist(), cols.detach().cpu().tolist()):
        key = ((int(row) // block_size) * block_size, (int(col) // block_size) * block_size)
        if key not in values:
            values.add(key)
            seen.append(key)
    return seen


def _source_block(torch, param, score, rows, cols, block_size: int, phi_source: str):
    dense = torch.zeros(block_size, block_size, device=param.device, dtype=param.dtype)
    local_rows = (rows - int(rows.min().item())).long()
    local_cols = (cols - int(cols.min().item())).long()
    if phi_source == "score":
        values = score[rows.detach().cpu(), cols.detach().cpu()].to(param.device, param.dtype)
    elif phi_source == "weight_score":
        score_values = score[rows.detach().cpu(), cols.detach().cpu()].to(param.device, param.dtype)
        values = param[rows, cols].detach() * score_values
    else:
        values = param[rows, cols].detach()
    dense[local_rows, local_cols] = values
    return dense.float()


def _svd_bases(torch, block, *, rank: int):
    rows, cols = block.shape
    used = max(1, min(rank, rows, cols))
    try:
        u, _s, vh = torch.linalg.svd(block, full_matrices=False)
        left = u[:, :used]
        right = vh[:used, :]
    except Exception:
        eye = torch.eye(max(rows, cols), device=block.device, dtype=block.dtype)
        left = eye[:rows, :used]
        right = eye[:used, :cols]
    if used < rank:
        pad_left = torch.zeros(rows, rank - used, device=block.device, dtype=block.dtype)
        pad_right = torch.zeros(rank - used, cols, device=block.device, dtype=block.dtype)
        left = torch.cat([left, pad_left], dim=1)
        right = torch.cat([right, pad_right], dim=0)
    return left, right


def _projected_basis(torch, left, basis, right):
    return torch.stack([left.matmul(item).matmul(right) for item in basis], dim=0)


def _phi_basis(torch, rank: int, variant: str, *, device, dtype):
    if variant == "diagonal":
        return _basis_from_indices(torch, rank, [(i, i) for i in range(rank)], device, dtype)
    if variant == "upper_triangular":
        return _basis_from_indices(
            torch,
            rank,
            [(i, j) for i in range(rank) for j in range(i, rank)],
            device,
            dtype,
        )
    if variant == "block_diagonal":
        pairs = []
        for start in range(0, rank, 2):
            for i in range(start, min(start + 2, rank)):
                for j in range(start, min(start + 2, rank)):
                    pairs.append((i, j))
        return _basis_from_indices(torch, rank, pairs, device, dtype)
    if variant == "kronecker":
        return _kronecker_basis(torch, rank, device=device, dtype=dtype)
    if variant == "hadamard":
        return _hadamard_basis(torch, rank, device=device, dtype=dtype)
    if variant == "codebook_2x2":
        return _codebook_basis(torch, rank, size=2, device=device, dtype=dtype)
    if variant == "codebook_4x4":
        return _codebook_basis(torch, rank, size=4, device=device, dtype=dtype)
    return _basis_from_indices(
        torch,
        rank,
        [(i, j) for i in range(rank) for j in range(rank)],
        device,
        dtype,
    )


def _basis_from_indices(torch, rank: int, pairs, device, dtype):
    values = []
    for row, col in pairs:
        item = torch.zeros(rank, rank, device=device, dtype=dtype)
        item[row, col] = 1.0
        values.append(item)
    return torch.stack(values, dim=0)


def _kronecker_basis(torch, rank: int, *, device, dtype):
    small = max(1, int(rank ** 0.5))
    while rank % small != 0 and small > 1:
        small -= 1
    other = max(1, rank // small)
    bases = []
    for a_i in range(small):
        for a_j in range(small):
            a = torch.zeros(small, small, device=device, dtype=dtype)
            a[a_i, a_j] = 1.0
            for b_i in range(other):
                for b_j in range(other):
                    b = torch.zeros(other, other, device=device, dtype=dtype)
                    b[b_i, b_j] = 1.0
                    bases.append(torch.kron(a, b)[:rank, :rank])
    return torch.stack(bases, dim=0)


def _hadamard_basis(torch, rank: int, *, device, dtype):
    identity = torch.eye(rank, device=device, dtype=dtype)
    ones = torch.ones(rank, rank, device=device, dtype=dtype)
    checker = torch.empty(rank, rank, device=device, dtype=dtype)
    for row in range(rank):
        for col in range(rank):
            checker[row, col] = 1.0 if (row + col) % 2 == 0 else -1.0
    return torch.stack([identity, ones, checker], dim=0)


def _codebook_basis(torch, rank: int, *, size: int, device, dtype):
    values = []
    for row in range(0, rank, size):
        for col in range(0, rank, size):
            item = torch.zeros(rank, rank, device=device, dtype=dtype)
            item[row:min(row + size, rank), col:min(col + size, rank)] = 1.0
            values.append(item)
    return torch.stack(values, dim=0)


def _phi_basis_count(rank: int, variant: str) -> int:
    if variant == "diagonal":
        return rank
    if variant == "upper_triangular":
        return rank * (rank + 1) // 2
    if variant == "block_diagonal":
        return sum(min(2, rank - start) ** 2 for start in range(0, rank, 2))
    if variant == "hadamard":
        return 3
    if variant in {"codebook_2x2", "codebook_4x4"}:
        size = 2 if variant == "codebook_2x2" else 4
        return ((rank + size - 1) // size) ** 2
    return rank * rank


__all__ = ["phi_candidate_summary", "phi_validation_indices"]
