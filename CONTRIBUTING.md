# Contributing to SAINT-G

SAINT-G means DRM por Enxerto com SAINT-G-Phi. The current technical definition is:

```text
sparse multi-scale block-codebook delta training
```

In Portuguese:

```text
treino de deltas esparsos por dicionario multi-escala de blocos
```

## Project Stage

SAINT-G is experimental. Contributions should preserve the research discipline of
the project: clear hypotheses, controlled baselines, reproducible benchmarks,
and honest success/failure criteria.

## Development Rules

- Keep Python files at 500 lines or fewer.
- Keep changes scoped to the phase or experiment being modified.
- Prefer dependency-free implementations unless a phase explicitly introduces a
  dependency such as PyTorch.
- Do not remove baselines just because SAINT-G loses against them.
- Record benchmark results in docs when they change the project verdict.
- Keep generated artifacts out of git unless they are intentionally small and
  useful as fixtures.

## Testing

Run the test suite before submitting changes:

```bash
python -m unittest discover -s tests
```

Also check the file line limit:

```bash
python -m unittest tests.test_file_line_limits
```

## Documentation

When changing architecture, runtime behavior, or phase status, update the
relevant files under `docs/`.

Important documents:

- `docs/paradigma_SAINT-G.md`
- `docs/arquitetura.md`
- `docs/roadmap.md`
- `docs/process/`

## Benchmarks

Benchmark changes should report:

- task or dataset;
- seed;
- method;
- trainable parameters;
- memory estimate;
- loss or reconstruction error;
- compression ratio when relevant;
- comparison against baselines.

## Pull Request Expectations

A good pull request should include:

- a short description of the change;
- tests or a clear explanation of why tests were not added;
- benchmark output when behavior changes;
- documentation updates when project conclusions change.
