# Phase 16 Marco 4O - Tensor-Network Follow-ups from ITensors.jl and NTK-Mirror

Status: **4O-lite SVD anatomy completed; 4O TT/MPS adapter baseline implemented; 4O-B capacity sanity follow-up completed and negative**.

This document records the technical ideas extracted from two external projects and maps them to concrete SAINT-G and DRM Transformer follow-up work:

- **ITensors.jl / ITensorMPS.jl**: tensor-network abstractions, named indices, SVD/truncation discipline, MPS/MPO methods, contraction-cost analysis, block-sparse tensor structure.
- **NTK-Mirror**: LoRA-free forward-pass adaptation through sparse signed activation log-gates, NTK-style gate selection, and log-gate composability.

The purpose is not to add heavy external dependencies to SAINT-G. The purpose is to preserve the useful algorithmic ideas and convert them into PyTorch-native diagnostics and baselines that fit the existing Phase 16 graft-routing pipeline.

## Executive Verdict

ITensors.jl is valuable for SAINT-G mostly as a source of tensor-network design patterns, not as a direct runtime dependency.

```text
Do not integrate Julia/ITensors.jl directly into the current Phase 16 loop.
Do borrow its tensor-network ideas and implement minimal PyTorch probes/baselines.
```

NTK-Mirror is closer to the current SAINT-G Phase 16 work because Marco 4M already implements an NTK-Mirror-inspired activation-gradient score:

```text
score(target) = sum(abs(grad_h * h))
```

However, SAINT-G should still treat NTK-Mirror as an inspiration and baseline family rather than a drop-in replacement. NTK-Mirror gates activation channels. SAINT-G graft blocks add trainable modules and validate staged model growth under a recomposable checkpoint protocol.

## Source Projects Reviewed

### ITensors.jl

Repository:

```text
https://github.com/ITensor/ITensors.jl
```

ITensors.jl is a Julia library for tensor-network algorithms. Its core abstractions are tensors with named/tagged indices whose semantics are separated from their memory layout. This makes tensor contraction and factorization less error-prone because index identity, tags, and prime levels carry structural information.

Important features for SAINT-G:

```text
- ITensor indices with tags and identities;
- dense and block-sparse tensor support;
- QN / conserved-quantity indexing;
- tensor contraction by index identity;
- contraction sequence cost and optimization;
- SVD, QR, eigen/factorization on arbitrary tensor index groupings;
- truncated SVD with cutoff, maxdim, mindim, and truncation-error reporting;
- GPU backends through Julia GPU packages, with dense CUDA/cuTENSOR support more mature than block-sparse GPU paths.
```

Since ITensors.jl v0.7, MPS/MPO functionality moved out of ITensors.jl into ITensorMPS.jl.

### ITensorMPS.jl

Repository:

```text
https://github.com/ITensor/ITensorMPS.jl
```

ITensorMPS.jl contains the finite MPS/MPO algorithms and types that are most directly relevant to tensor-train adapter baselines:

```text
- MPS and MPO types;
- random_mps;
- OpSum -> MPO construction;
- DMRG;
- truncate!;
- maxlinkdim;
- inner / dot / loginner;
- apply and gate evolution utilities.
```

For SAINT-G, the most important transferable idea is not DMRG itself. It is the controlled representation of high-order objects as low-bond-dimension tensor networks, with explicit truncation and error accounting.

### NTK-Mirror

Repository:

```text
https://github.com/leochlon/ntkmirror
```

NTK-Mirror learns sparse signed log-gates on frozen Transformer decoder-layer output channels:

```text
h'_{layer, token, channel} = exp(s_{layer, channel}) * h_{layer, token, channel}
```

Gate selection is based on a log-gate derivative:

```text
dL/ds_{l,c} = sum_t <dL/dh_{l,t,c}, h_{l,t,c}>
```

The public package fits small controllers, saves them as `controller.pt`, attaches them during evaluation/generation, and composes task controllers in signed log-gate coordinates:

```text
s_AB = clip(w_A * s_A + w_B * s_B, -max_log_gate, max_log_gate)
h'   = exp(s_AB) * h
```

This directly inspired SAINT-G Marco 4M's activation-gradient target score. The difference is that Marco 4M currently uses the score as a diagnostic/routing signal for graft targets, while NTK-Mirror trains the activation gates themselves as the adaptation mechanism.

## Why Direct ITensors.jl Integration Is Not Recommended Now

SAINT-G and DRM Transformer are currently PyTorch projects. Adding ITensors.jl would introduce:

```text
- a Julia runtime dependency;
- cross-language tensor conversion complexity;
- duplicated GPU-memory management concerns;
- a separate package/environment story;
- extra failure modes during long CUDA benchmark runs;
- little immediate benefit for Phase 16 routing, which can use PyTorch SVD and hooks directly.
```

The current practical path is:

```text
ITensors.jl idea -> PyTorch implementation -> SAINT-G report/run artifact
```

Only revisit a Julia bridge if a later experiment specifically requires ITensorMPS.jl algorithms that are hard to reproduce in PyTorch.

## Relationship to Current Phase 16 State

The current Phase 16 sequence established:

```text
4H: best prior fine-grained second-stage result, 5 grafts.
4I: residual/orthogonal routing variant, 5 grafts, slightly worse than 4H.
4J: two-pass candidate pruning, valid CUDA run, but accepted only 4 grafts.
4K: larger top-k/probe variant documented to recover missed useful targets.
4L: multiseed robustness follow-up for 4K-style behavior.
4M: implemented NTK-Mirror-inspired activation-gate probe.
4N: planned NTK-guided candidate pruning/routing if 4M is predictive.
```

The ITensor-inspired work should come after the 4M/4N decision, because it answers a different question:

```text
After the routing chooses useful grafts, how much structured capacity do those grafts actually need?
```

That makes the next recommended tensor-network follow-up:

```text
Marco 4O-lite - Graft SVD Anatomy
```

4O-lite has now been executed and documented:

```text
docs/reports/phase16_marco4o_lite_graft_svd_anatomy.md
runs/phase16_marco4o_lite_graft_svd_seed42_seed7_seed123_effective/graft_svd_anatomy.md
```

Short result: accepted `up` matrices remain full-rank-like (`rank@99% = 95/96` on average), accepted `down` matrices are moderately concentrated (`rank@99% ~= 19.8`), and the linearized diagnostic product `up @ down` is strongly low-rank-looking (`rank@99% ~= 2.8`). Therefore 4O-lite supports a controlled low-rank baseline sweep, not blind truncation of the current full graft blocks.

and then, only if compression signals are strong:

```text
Marco 4O - Tensor-Train / MPS Adapter Baseline
```

## Idea 1 - Graft SVD Anatomy

### Motivation

ITensors.jl emphasizes factorization and controlled truncation. A direct SAINT-G translation is to analyze trained graft weights by singular spectrum and truncation error.

The key question:

```text
Are accepted graft blocks using their full capacity, or are they mostly low-rank?
```

If accepted grafts are highly compressible, SAINT-G can use smaller low-rank or tensor-train adapters. If not, graft-block capacity is justified, and the spectra become a diagnostic rather than a compression path.

### Proposed Marco

```text
Marco 4O-lite - Graft SVD Anatomy
```

### Scope

Create a script:

```text
scripts/analyze_phase16_graft_svd.py
```

Inputs:

```text
--checkpoint /path/to/composed_graft_checkpoint.pt
--stage-metrics /path/to/stage_metrics.json
--candidate-metrics /path/to/candidate_metrics.json
--output-dir /path/to/svd_report
```

Outputs:

```text
svd_metrics.json
singular_spectra.jsonl
rank_summary.csv
results.md
plots/*.png        # optional if matplotlib is available
```

### Metrics

For every trained/accepted graft matrix that can be extracted:

```text
- matrix name;
- shape;
- parameter count;
- Frobenius norm;
- singular values;
- cumulative energy curve;
- effective_rank_99;
- effective_rank_999;
- effective_rank_9999;
- stable rank = ||W||_F^2 / ||W||_2^2;
- entropy effective rank;
- truncation error at ranks 1, 2, 4, 8, 16, 32, 64;
- reconstruction error if the matrix is truncated and reinserted offline.
```

Recommended truncation-error formula:

```text
relative_truncation_error(k) = sum_{i>k} sigma_i^2 / sum_i sigma_i^2
```

This mirrors ITensors.jl's truncated-SVD discipline while staying in PyTorch.

### Comparisons

Compare spectra across:

```text
- accepted grafts vs rejected candidates if candidate checkpoints exist;
- seed 42 vs seed 7 vs seed 123;
- targets blocks.2 / blocks.3 / blocks.4;
- early accepted grafts vs the fifth graft;
- runs that find the fifth graft vs runs that stop at four grafts.
```

### Success Criteria

Marco 4O-lite is useful if it yields at least one actionable result:

```text
1. accepted grafts are low-rank enough to justify compression;
2. accepted and rejected grafts have visibly different spectra;
3. the fifth graft has a distinct rank/energy signature;
4. seed-robust runs show more stable spectra than seed-fragile runs;
5. truncation preserves loss within an acceptable tolerance.
```

### Failure Criteria

```text
- spectra are flat and not compressible;
- accepted and rejected grafts look indistinguishable;
- extracting matrices from checkpoints is unreliable;
- offline truncated reconstruction changes composed loss too much;
- analysis cost is high compared to insight gained.
```

## Idea 2 - Tensor-Train / MPS Adapter Baseline

### Motivation

ITensorMPS.jl suggests representing large structured transformations as chains of smaller tensors with a bounded internal bond dimension. In machine-learning terms, this maps to Tensor Train (TT) or MPS-style layers.

For SAINT-G, the baseline would ask:

```text
Can a low-bond-dimension tensor-network adapter match or beat graft blocks under the same routing protocol?
```

### Proposed Marco

```text
Marco 4O - Tensor-Train / MPS Adapter Baseline
```

### Design

Implement a PyTorch-native adapter, not a Julia bridge:

```text
TTLinear
TTGraftBlock
```

Candidate structure:

```text
input x: [batch, seq, d_model]
reshape channel axis into tensorized dimensions: d_model = prod(input_dims)
apply TT/MPS factorized transform with bond_dim chi
reshape back to [batch, seq, d_model]
add residual adapter output into the target block
```

For DRM's current 5M-scale configuration, `d_model` may not factor nicely. The implementation should support padding or a learned projection into a tensorized adapter width:

```text
x -> project_down -> TT/MPS core transform -> project_up -> residual add
```

This avoids forcing `d_model` itself to have a convenient factorization.

### Sweep

Start with small bond dimensions:

```text
bond_dim / chi: 2, 4, 8, 16
adapter_width: 64, 128, 256 if projection is used
activation: silu
initialization scale: match current graftblock candidates
```

### Comparison Axes

Compare against graft blocks under the existing Phase 16 routing protocol:

```text
- composed_loss;
- accepted_grafts;
- validation gain per parameter;
- validation gain per byte of checkpoint;
- training time;
- peak GPU memory;
- recompose_abs_diff;
- seed robustness;
- whether the fifth useful adaptation appears.
```

### Success Criteria

```text
- TT/MPS adapter reaches similar loss with fewer trainable parameters;
- TT/MPS adapter gives smaller checkpoint artifacts;
- TT/MPS adapter is more seed-robust than graft blocks;
- TT/MPS adapter recovers the fifth graft behavior without larger candidate probes.
```

### Failure Criteria

```text
- TT/MPS adapter underfits at practical bond dimensions;
- training is slower due to inefficient tensor contractions;
- parameter savings do not translate into better loss/byte/time;
- routing behavior becomes less stable than graft blocks.
```

## Idea 3 - Tensor-Network Cost as a Routing Feature

ITensors.jl includes contraction-cost analysis and contraction sequence optimization. SAINT-G can borrow this idea at the routing level.

Current routed candidates are mostly evaluated by validation gain and penalties. A future 4N/4P score could include an estimated capacity/runtime/memory term:

```text
candidate_score = gain
                - lambda_runtime * estimated_runtime_cost
                - lambda_memory  * estimated_memory_cost
                - lambda_params  * trainable_parameter_cost
                + alpha_ntk      * ntk_score_norm
```

The goal is not to compute an exact tensor-network contraction plan for PyTorch modules. The goal is to make routing aware that two candidates with similar gain may have different downstream cost profiles.

### Candidate Features

```text
- target block index;
- candidate parameter count;
- estimated activation memory;
- estimated FLOPs per forward/backward;
- adapter checkpoint bytes;
- NTK activation score;
- target redundancy/overlap with existing accepted grafts;
- SVD effective-rank signature if available.
```

### Practical Use

This would become useful after 4M/4O-lite because those reports can supply additional routing features:

```text
4M -> activation-gradient sensitivity
4O-lite -> compressibility / rank signature
4O -> cost of TT/MPS adapter family
```

## Idea 4 - BlockSparse/QN-Inspired Sparse Channel Routing

ITensors.jl has QN and BlockSparse tensors, where tensor blocks are present only when index-sector constraints allow them. SAINT-G can use this as a design analogy for sparse channel routing.

Potential SAINT-G translation:

```text
- partition hidden channels into groups/sectors;
- assign grafts to sectors instead of the full hidden axis;
- discourage overlap between accepted graft sectors;
- use NTK channel scores to identify active sectors;
- track sector collisions across seeds and stages.
```

This connects ITensors.jl and NTK-Mirror:

```text
NTK-Mirror identifies sensitive layer-channel directions.
ITensors suggests representing selected directions as structured sparse sectors.
```

Possible future adapter:

```text
SectorGraftBlock:
  only applies to selected channel groups;
  stores sector metadata in checkpoint;
  composes without touching unrelated sectors.
```

Success signal:

```text
Sector routing reduces interference between grafts and improves seed robustness.
```

## Idea 5 - DRM Transformer Manifold Attention Tensor Anatomy

For `/mnt/e/dev/ai/drm_transformer`, ITensor-style thinking is most useful as an analysis lens.

The DRM Transformer block currently has:

```text
x = x + attn(norm1(x), metric_net, gravity_field, anchors)
x = x + ffn(norm2(x))
```

The attention path contains structured tensors:

```text
- token embeddings x;
- per-head/query/key/value tensors;
- manifold coordinates;
- metric G(x);
- anchor coordinates;
- gravity terms;
- attention logits and probabilities.
```

### Proposed DRM Diagnostic

```text
DRM Marco A - Manifold Attention Tensor Anatomy
```

### Scope

Add an offline analysis script in the DRM Transformer repo:

```text
scripts/analyze_manifold_attention_tensor_anatomy.py
```

Inputs:

```text
--checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/.../final.pt
--data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m
--device cuda
--batches N
--seq-len 128
--output-dir /path/to/drm_tensor_anatomy
```

Outputs:

```text
tensor_anatomy_metrics.json
layer_head_spectra.jsonl
rank_summary.csv
results.md
```

### Tensor Views to Analyze

```text
[layer, head, token, manifold_dim]
[batch * token, head * manifold_dim]
[token, token, head]
[layer, head, token, token]
[layer, token, anchor]
```

### Metrics

```text
- effective rank by layer/head;
- singular spectrum of attention logits/probabilities;
- compressibility of manifold coordinates;
- rank of metric-induced transformations;
- correlation between gravity terms and attention rank;
- layer/head redundancy;
- whether late layers use higher-dimensional structure than early layers.
```

### Why This Matters

If DRM's geometric attention is low-rank or sectorized, it can guide:

```text
- cheaper inference kernels;
- targeted graft placement;
- better pruning/routing features;
- compression of metric/gravity modules;
- selection of which layers deserve SAINT-G growth.
```

## How NTK-Mirror and ITensors.jl Fit Together

The two repos point at complementary forms of structure:

```text
NTK-Mirror:
  local activation/channel sensitivity and cheap forward-pass adaptation.

ITensors.jl / ITensorMPS.jl:
  global tensor factorization, contraction cost, low-bond-dimension structure,
  and truncation-error accounting.
```

In SAINT-G terms:

```text
4M/4N answer: where should adaptation happen?
4O-lite answers: how much rank/capacity did useful adaptation need?
4O answers: can that capacity be represented more cheaply as TT/MPS?
```

4O has now been executed in smoke form:

```text
docs/reports/phase16_marco4o_tt_mps_adapter_baseline.md
runs/phase16_marco4o_tt_mps_adapter_seed42_chi{2,4,8,16}_smoke/
```

Short result: the PyTorch-native TT/MPS adapter family is implemented and recomposes exactly, but the seed-42 width-128 chi 2/4/8/16 smoke sweep produced zero composed gain and accepted zero grafts. This is a negative first modeling result, not an infrastructure failure.


A future combined routing score could look like:

```text
candidate_score = composed_gain
                + alpha * normalized_ntk_sensitivity
                + beta  * compressibility_prior
                - gamma * estimated_cost
                - delta * overlap_with_existing_grafts
```

## Recommended Order of Work

Completed through first 4O smoke execution:

```text
1. Marco 4M/4N decision path completed through conservative 4N-B / 4N-C analysis.
2. Marco 4O-lite Graft SVD Anatomy completed.
3. Marco 4O TT/MPS Adapter Baseline implemented and smoke-executed.
```

Current recommended next step:

```text
4. Marco 4O-B TT/MPS Capacity Sanity Sweep completed:
   - adapter_width 256/512;
   - chi 4/8/16;
   - longer stage-1 candidate probes;
   - dense parameter-matched baseline.
5. Since 4O-B remained zero-gain, deprioritize TT/MPS and move to cost-aware dense graft routing.
6. In parallel or later, run DRM Marco A tensor anatomy on drm_transformer.
```

## Minimal Commands to Add Later

### 4O-lite Sketch

```bash
python \
  scripts/analyze_phase16_graft_svd.py \
  --checkpoint /path/to/composed_graft_checkpoint.pt \
  --stage-metrics /path/to/stage_metrics.json \
  --candidate-metrics /path/to/candidate_metrics.json \
  --output-dir /path/to/phase16_marco4o_lite_svd
```

### DRM Tensor Anatomy Sketch

```bash
python \
  scripts/analyze_manifold_attention_tensor_anatomy.py \
  --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
  --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
  --device cuda \
  --batches 16 \
  --seq-len 128 \
  --output-dir /path/to/drm_tensor_anatomy
```

## Prior-Art Links

- Fishman, White, and Stoudenmire, **The ITensor Software Library for Tensor Network Calculations**, SciPost Phys. Codebases 4, 2022. DOI: `10.21468/SciPostPhysCodeb.4`.
- ITensors.jl: `https://github.com/ITensor/ITensors.jl`.
- ITensorMPS.jl: `https://github.com/ITensor/ITensorMPS.jl`.
- Chlon, **NTK-Mirror: LoRA-free forward-pass fine-tuning via signed log-mask controllers**, 2026. `https://github.com/leochlon/ntkmirror`.

## Open Questions

```text
1. Does Marco 4M's NTK ranking align with the target selected by deep candidate training?
2. Is the fifth useful graft spectrally different from the first four grafts?
3. Are accepted grafts compressible enough for low-rank or TT/MPS replacement?
4. Does a tensor-network adapter improve gain per byte/time compared with graft blocks?
5. Does DRM geometric attention show low-rank or sectorized structure by layer/head?
6. Can NTK channel sensitivity and SVD compressibility be combined into a better router?
```
