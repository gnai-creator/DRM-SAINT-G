# Phase 16 Marco 4N - NTK-Guided Candidate Pruning and Routing

Status: **planned / dependent on Marco 4M diagnostic result**.

## Goal

Use the NTK-Mirror-inspired activation-gate score from Marco 4M as an automatic
candidate pruning/routing signal for staged graft growth.

Marco 4M only records:

```text
score(block) = sum(abs(grad_h * h))
```

Marco 4N promotes that score into routing if 4M shows that it predicts the useful
stage targets.

## Motivation

The current two-pass candidate pruning recipe is expensive:

```text
probe every target/lr/scale/activation candidate
sort by composed_gain_orthogonal
run deep training on top-k candidates
```

For the 4K/4L grid this means:

```text
3 targets * 3 lrs * 3 scales * 1 activation = 27 probe candidates per stage
then top-k=8 deep candidates per stage
```

An NTK-guided prefilter could reduce this by ranking targets before candidate
training.

## Proposed Modes

Implement one or both modes after 4M validation:

```text
ntk_prefilter:
  rank target modules with NTK activation score
  keep only top target(s)
  run the existing lr/scale candidate grid on those targets

ntk_score_blend:
  keep existing candidate probe
  add normalized NTK target score into candidate_score
  candidate_score = composed_gain - orthogonal_penalty + alpha * ntk_score_norm
```

## Candidate CLI Shape

Proposed future flags:

```text
--candidate-score-mode ntk_prefilter
--ntk-prefilter-top-targets 1
--ntk-score-alpha 0.0
```

or:

```text
--candidate-score-mode ntk_score_blend
--ntk-score-alpha 0.0001
```

The exact names can change during implementation, but the principle should stay:
4N must be an automatic routing experiment, not just a diagnostic artifact.

## Success Criteria

Marco 4N passes if NTK-guided pruning/routing does at least one of the following:

```text
- preserves or improves the 4K seed-42 result with fewer candidate probes;
- recovers the fifth graft on a seed where 4L rejected stage 2;
- reduces total runtime while matching 4K/4L composed_loss within tolerance;
- improves stage-2 target selection consistency across seeds.
```

## Failure Criteria

Marco 4N fails if:

```text
- NTK target ranking does not correlate with useful deep candidates;
- NTK prefilter drops the target later selected by deep training;
- blended scoring makes seed robustness worse;
- runtime overhead from the NTK pass exceeds candidate-probe savings.
```

## Relationship to NTK-Mirror

This is not a direct port of `ntkmirror`. SAINT-G still uses graft blocks.

The imported idea is only the activation-gate diagnostic:

```text
dL/ds = grad_h * h
```

If this signal proves predictive, a later phase can test a true NTK-Mirror-style
small activation controller as a separate cheap adapter baseline against graft
blocks.
