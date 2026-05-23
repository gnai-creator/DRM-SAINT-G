# Prior Art

SAINT-G is an experimental research system for structured model growth by
grafting. Its current direction evolved from sparse multi-scale block-codebook
delta training toward compact grafts of the form:

```text
Delta W = A Phi B
```

This document exists to keep novelty claims conservative. SAINT-G should be
evaluated as a combination and extension of known ideas unless experiments prove
otherwise.

## Related Areas

SAINT-G overlaps with:

- parameter-efficient fine-tuning;
- adapters and adapter fusion;
- LoRA and low-rank adaptation;
- QLoRA and quantized adapter training;
- DoRA and weight-decomposed adaptation;
- VeRA and shared random projection adapters;
- LoKr/Kronecker adapters;
- LoHa/Hadamard product adapters;
- IA3-style multiplicative adaptation;
- sparse fine-tuning;
- block-sparse updates;
- structured matrix factorization;
- tensor/Kronecker/Hadamard decompositions;
- vector quantization and product quantization;
- codebook learning;
- residual quantization;
- model compression and pruning;
- quantization-aware training;
- delta checkpoints;
- model merging and task arithmetic;
- progressive networks and modular growth;
- mixture-of-experts routing;
- neural architecture growth and dynamic capacity allocation.

## Important Baselines

SAINT-G experiments should compare against:

- frozen base model;
- full fine-tuning;
- full-module fine-tuning under the same parameter budget;
- head-only tuning;
- LoRA with tuned rank and learning rate;
- QLoRA when quantized baselines are feasible;
- DoRA/VeRA/LoKr/LoHa-style baselines when implemented;
- low-rank matrix approximation;
- SVD-initialized adapters;
- uniform quantization;
- block codebook reconstruction;
- budgeted full delta;
- block-budgeted delta;
- random sensitivity maps;
- activation and gradient routing controls.

## Known Risk

Many SAINT-G ideas may be rediscovering or recombining existing techniques.
The project should treat novelty as a hypothesis, not an assumption.

Specific risks:

- `A Phi B` can look like a generalized LoRA family unless `Phi` provides a
  demonstrably useful structured operator.
- Codebook and block reuse overlap with vector quantization and product
  quantization.
- Routing by sensitivity overlaps with pruning, sparse training and MoE routing.
- Compact checkpoints overlap with adapter and delta-checkpoint literature.
- Progressive grafting overlaps with modular growth and progressive networks.

## Current SAINT-G Distinction

The working distinction is not "low-rank adaptation" alone and not merely
"compress a matrix." The current hypothesis is:

```text
route where capacity should grow, train compact structured grafts such as
A Phi B, validate each graft by gain per parameter/byte/time, and keep the model
recomposable through checkpointed graft artifacts and optional consolidation.
```

This distinction is still experimental. It becomes meaningful only if benchmarks
show advantages against tuned PEFT, dense and budgeted baselines on at least one
important axis:

- validation loss;
- gain per trainable parameter;
- checkpoint size;
- memory peak;
- routing/training time;
- retention after consolidation;
- scalability of candidate search.

## Claim Discipline

Acceptable claims:

- SAINT-G is experimental.
- SAINT-G tests structured grafting as a model-growth method.
- SAINT-G has internal benchmarks where specific variants beat specific
  baselines under stated budgets.
- SAINT-G does not currently claim general superiority over LoRA/QLoRA.
- 70B support is a roadmap target for partial adaptation, not full pretraining.

Avoid these claims until proven:

- "SAINT-G trains 70B on a consumer GPU."
- "SAINT-G is better than LoRA."
- "SAINT-G replaces dense pretraining."
- "SAINT-G is a new paradigm" without qualifying it as a research
  hypothesis.
- "5M grafted equals 350M full" without defining the comparison axis.

## How to Use This Document

When a new experiment is added, update this file if it becomes clearly related
to known prior work. When SAINT-G loses to an existing method, keep that
result in the record.
