# SAINT-G

<p align="center">
  <img src="assets/saint-g-banner.png" alt="SAINT-G banner" width="800">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg" alt="License: AGPL-3.0"></a>
  <a href="LICENSE-COMMERCIAL.md"><img src="https://img.shields.io/badge/License-Commercial-orange.svg" alt="Commercial License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-3776ab.svg" alt="Python 3.10+"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/PyTorch-2.11%2B-ee4c2c.svg" alt="PyTorch 2.11+"></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/Runtime-SAINT--G-blueviolet.svg" alt="Runtime SAINT-G"></a>
  <a href="#core-idea"><img src="https://img.shields.io/badge/Grafting-A%20Phi%20B-green.svg" alt="Grafting A Phi B"></a>
  <a href="https://github.com/gnai-creator/drm_transformer"><img src="https://img.shields.io/badge/Backbone-drm__transformer-orange.svg" alt="Backbone drm_transformer"></a>
  <a href="docs/roadmap.md"><img src="https://img.shields.io/badge/Roadmap-Controlled%20AI%20Growth-yellow.svg" alt="Roadmap Controlled AI Growth"></a>
  <a href="#scalability"><img src="https://img.shields.io/badge/Cluster-Online%20graft%20search-0f766e.svg" alt="Online graft search"></a>
</p>

<p align="center">
  <strong>Scalable Auditable Intelligence through Neural Grafting</strong>
</p>

<p align="center">
  <em>A framework for controlled, modular, and auditable AI growth.</em>
</p>

<p align="center">
  <a href="docs/roadmap.md">Roadmap</a> |
  <a href="docs/process/">Process Docs</a> |
  <a href="CONTRIBUTING.md">Contributing</a> |
  <a href="SECURITY.md">Security</a>
</p>

SAINT-G is a research framework for growing AI systems through small,
validated, recomposable neural grafts instead of opaque monolithic retraining.

The long-term thesis is simple:

```text
The safer path to more capable AI may not be only making models larger.
It may be making growth modular, testable, reversible, and governed.
```

SAINT-G is designed around a different unit of progress:

```text
base model
  + candidate grafts
  + validation gates
  + retention checks
  + safety checks
  + rollback
  + audit trail
  + periodic consolidation
```

The first backbone is
[`drm_transformer`](https://github.com/gnai-creator/drm_transformer), a custom
geometric Transformer based on Directional Relational Manifolds. The first
growth method is DRM grafting, but the broader project is now SAINT-G:
Scalable Auditable Intelligence through Neural Grafting.

---

## Index

- [Why This Exists](#why-this-exists)
- [Core Idea](#core-idea)
- [SAINT-G vs Traditional Training](#saint-g-vs-traditional-training)
- [Architecture](#architecture)
- [Current Research Stage](#current-research-stage)
- [Quick Start](#quick-start)
- [DRM Transformer Bridge](#drm-transformer-bridge)
- [Scalability](#scalability)
- [Continual Growth](#continual-growth)
- [What This Does Not Claim](#what-this-does-not-claim)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why This Exists

Today, model improvement is usually treated as a dense training problem:
update huge tensors, store huge optimizer state, publish a new monolithic
checkpoint, and hope the behavioral changes are acceptable.

That works, but it is hard to audit.

SAINT-G explores another path:

```text
freeze most of the model
find where growth may help
train compact graft candidates
validate them against the composed model
accept only what improves real metrics
keep every change removable and traceable
```

The goal is not merely parameter efficiency. The goal is controlled growth:

- every graft has metadata, metrics, hashes, and provenance;
- every accepted change can be recomposed and evaluated;
- every risky or regressive graft can be removed;
- every consolidation step can be audited;
- every gain is compared against strong baselines.

## Core Idea

The current strongest technical object is a neural graft:

```text
Delta W = A Phi B
```

Where:

- `W` is a frozen target matrix or module;
- `A` projects into the graft space;
- `Phi` is the compact trainable operator;
- `B` projects back to the target space;
- `Delta W` is applied by hook, sparse update, or consolidation.

In the DRM experiments, grafts are trained, validated, accepted/rejected, and
stored as recomposable artifacts.

Variants explored so far include:

- dense Phi;
- diagonal Phi;
- upper triangular Phi;
- Hadamard Phi;
- low-rank Phi;
- least-squares initialized Phi;
- Phi with sparse residual;
- trainable `A/B` under a parameter cap;
- staged graft growth;
- validation-routed graft selection;
- fine-grained second-stage growth.

## SAINT-G vs Traditional Training

| Component | Traditional full training | LoRA/QLoRA | SAINT-G |
|---|---|---|---|
| Base weights | updated | frozen or quantized | frozen by default |
| Trainable object | full tensors | low-rank adapter | validated graft |
| Delta shape | dense | low-rank | structured `A Phi B` / graft block |
| Selection | all layers or manual | target modules | routing + validation gates |
| Acceptance | final training objective | adapter validation | composed-model validation |
| Checkpoint | full model or adapter | adapter | graft artifact + registry metadata |
| Growth | fixed retraining run | task adaptation | progressive, reversible growth |
| Auditability | low | medium | design goal |

SAINT-G does not assume it beats LoRA or QLoRA. Those are required baselines.
The project advances only where SAINT-G shows an advantage in at least one
serious axis: memory, checkpoint size, gain per parameter, reversibility,
validation-gated growth, or auditability.

## Architecture

```text
        data / evals / safety checks
                    |
                    v
          +--------------------+
          | sensitivity maps   |
          +--------------------+
                    |
                    v
          +--------------------+
          | candidate router   |
          +--------------------+
                    |
                    v
 frozen base ---- target layer/module ---- candidate grafts
                    |
                    v
          +--------------------+
          | train graft        |
          +--------------------+
                    |
                    v
          +--------------------+
          | composed validation|
          +--------------------+
                    |
          accept / reject / defer
                    |
                    v
          +--------------------+
          | graft registry     |
          +--------------------+
                    |
                    v
          +--------------------+
          | rollback / merge   |
          +--------------------+
```

Main modules:

```text
saint/
  adapters/       DRM, Hugging Face, graft application
  blocks/         block partitioning and reconstruction
  checkpoints/    compact/sharded payloads and checksums
  codebook/       block dictionaries and reuse
  memory/         memory estimation and dtype planning
  routing/        budget, sensitivity, validation rerank
  sensitivity/    gradient, Fisher, activation and proxy maps
  training/       toy tasks, linear tasks, mini-transformer tasks
  cli/            runtime commands
```

## Current Research Stage

SAINT-G has moved through several layers of validation:

- traditional LLM training paradigm documentation;
- block-codebook reconstruction;
- routed sparse delta training;
- linear-layer learning benchmarks;
- mini-transformer experiments;
- sensitivity maps;
- robust and scalable checkpoint formats;
- Hugging Face small-model bridge;
- 3B and 14B partial adaptation probes;
- DRM progressive grafting;
- Phi/graft variants;
- full DRM 125M smoke baseline;
- DRM 5M + grafted-to-125M comparison path.

The current bridge is:

```text
DRM full 125M/350M
vs
DRM 5M + SAINT-G grafted
vs
GPT-2/OPT size-band calibration
```

Recent Phase 16 results showed that staged grafting can produce small but real
validation gains with exact recomposition:

```text
base DRM 5M
  -> 4 accepted grafts
  -> fine-grained G2 accepted
  -> checkpoint recomposes with zero drift
```

This does not mean a 5M model has reached full 125M quality. It means the
growth path is operational and measurable.

## Quick Start

Create an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the CLI:

```powershell
python -m saint.cli --help
```

Run tests:

```powershell
python -m pytest
```

Inspect a small runtime command:

```powershell
python -m saint.cli estimate --help
```

## DRM Transformer Bridge

The current full-model comparison uses real `drm_transformer` scaling configs:

```text
configs/scaling/multilingual/125m.yaml
configs/scaling/multilingual/350m.yaml
```

Prepare the 350M dataset once:

```powershell
python scripts/prepare_multilingual_data.py `
  --output-dir data/multilingual_350m `
  --max-tokens 7000000000 `
  --vocab-size 50000 `
  --langs en,pt,es,fr,de
```

Finalize and clean raw shards:

```powershell
python scripts/prepare_multilingual_data.py `
  --output-dir data/multilingual_350m `
  --vocab-size 50000 `
  --finalize --clean-raw
```

Derive the 125M dataset:

```powershell
python scripts/prepare_multilingual_data.py `
  --derive-subset-from data/multilingual_350m `
  --output-dir data/multilingual_125m `
  --max-tokens 3500000000 `
  --subset-copy-mode hardlink
```

Smoke test the full 125M DRM:

```powershell
python scripts/train_distributed.py `
  --config configs/scaling/multilingual/125m.yaml `
  --device cuda `
  --override batch_size=1 gradient_accumulation_steps=8 total_tokens=819200 save_interval=100 eval_interval=100 log_interval=10 save_dir=checkpoints/multilingual_125m/smoke_100
```

## Scalability

SAINT-G is designed to scale in two ways.

### Single GPU

On a consumer GPU, the priority is controlled memory:

- frozen base model;
- micro-batch 1;
- sparse or compact deltas;
- checkpoint payloads that avoid dense materialization;
- routed training instead of full updates;
- cheap validation before expensive consolidation.

### GPU Cluster

On a cluster, the main opportunity is parallel graft search:

- GPU 1 tests graft candidates for layer A;
- GPU 2 tests graft candidates for layer B;
- GPU 3 runs LoRA/dense controls;
- GPU 4 validates old examples for regression;
- a coordinator approves, rejects, defers, or retries grafts.

This is not one huge synchronized dense run. It is distributed search for useful
growth modules.

```text
base model frozen
        |
        v
workers train candidate grafts
        |
        v
central validator measures composed gain
        |
        v
accept / reject / defer
        |
        v
recomposable checkpoint
```

## Continual Growth

If validation-gated grafting works at 125M/350M and later at cluster scale,
SAINT-G becomes a continual growth system:

```text
base model
  + verified graft registry
  + distributed graft search
  + continual safety gates
  + rollback
  + distillation
  + governance layer
```

Planned components:

- **Graft Registry:** versioned metadata, datasets, evals, hashes, compatibility.
- **Rollback:** remove one graft without discarding the whole model.
- **Graft Distillation:** consolidate many grafts into a new compact base.
- **Safety-Gated Growth:** quality, retention, safety, interpretability, conflict, rollback gates.
- **Specialized Graft Libraries:** code, math, Portuguese, legal, medical, safety, tool use.
- **Auditable Composition:** identify which graft changed which metric or behavior.
- **Governed Self-Improvement:** candidates can be proposed automatically, but accepted only through external validation and policy gates.

The larger research question:

```text
Can an AI system improve continuously without losing traceability,
correctability, and control?
```

## What This Does Not Claim

SAINT-G does not currently claim:

- full 70B pretraining on a consumer GPU;
- universal superiority over LoRA/QLoRA;
- replacement for dense pretraining;
- proof that grafting beats full training in general;
- autonomous self-modification without governance.

The honest claim is narrower:

```text
SAINT-G is a research system for testing controlled AI growth through
small, validated, auditable, and reversible neural grafts.
```

## Roadmap

Near-term:

1. Finish the full DRM 125M/350M vs grafted comparison.
2. Replicate with more seeds, splits, and at least one additional config.
3. Compare against stronger LoRA/QLoRA/full-module/sparse baselines.
4. Add retention, regression, and safety/control evals.
5. Formalize the DRM-Growth Protocol.
6. Prototype DRM-GOS: distributed validation-gated graft search.

Long-term:

- 1.3B bridge before 70B;
- 70B partial adaptation with quantized/frozen base;
- cluster-scale online graft search;
- graft registry and rollback;
- continual safety-gated growth;
- distillation of accumulated grafts;
- publication-quality reports.

Full roadmap:

```text
docs/roadmap.md
docs/process/
```

## License

SAINT-G is available under a dual-license model:

- **AGPL-3.0** for open-source use compatible with AGPL obligations.
- **Commercial license** for proprietary, closed-source, SaaS, OEM, or other
  deployments that need different terms.

For commercial licensing, contact `felipe@truthagi.ai`.

See:

- `LICENSE`
- `LICENSE-COMMERCIAL.md`
- `COPYRIGHT`
- `CLA.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `PRIOR_ART.md`
