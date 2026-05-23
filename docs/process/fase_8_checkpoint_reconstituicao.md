# Fase 8 - Checkpoint e Reconstituicao

Status: **concluida**.

Esta fase transforma o checkpoint do runtime em um artefato recomponivel, nao
apenas um registro de metricas.

## Objetivo

Salvar deltas reais e reconstruir pesos mesclados a partir de:

```text
modelo base + delta_payload -> merged_weights
```

## Implementado

- `checkpoint.json` agora salva `delta_payload` quando o metodo produz deltas;
- `metrics.json` indica `has_delta_payload`;
- `resume` valida que o payload de delta existe quando esperado;
- `merge` carrega `config.json`, recria o modelo base e aplica os deltas;
- `merged.json` passa a conter `merged_weights`.
- `pyproject.toml` declara o entrypoint `SAINT-G`;
- o runtime ganhou um adapter inicial `drm_transformer` para inspecionar matrizes
  2D de checkpoints e servir pesos-base para reconstituicao.

## Arquivos

```text
saint/checkpoints/manager.py
saint/runtime/runner.py
saint/transformer/SAINT-G_adapter.py
saint/adapters/drm_transformer.py
pyproject.toml
tests/test_runtime_phase7.py
```

## Comandos

```bash
python -m saint.cli train --config configs/runtime_smoke.json
python -m saint.cli resume --run runs/runtime_smoke
python -m saint.cli merge --run runs/runtime_smoke
```

Quando o pacote estiver instalado, os comandos equivalentes sao:

```bash
SAINT-G train --config configs/runtime_smoke.json
SAINT-G resume --run runs/runtime_smoke
SAINT-G merge --run runs/runtime_smoke
```

## Artefatos

```text
runs/runtime_smoke/checkpoint.json
runs/runtime_smoke/merged.json
```

Campos importantes:

```text
checkpoint.json:
  has_delta_payload: true
  delta_payload: {...}

merged.json:
  merged: true
  merged_weights: {...}
```

## Criterio de Conclusao

A fase conclui quando:

- checkpoint salva deltas reais;
- resume valida o checkpoint;
- merge reconstroi pesos a partir do modelo base e dos deltas;
- testes automatizados cobrem o fluxo;
- o runtime pequeno continua executando de ponta a ponta.

## Veredito

```text
Fase 8 concluida para o mini-transformer dependency-free.
```

## Pendencias Para Fases Futuras

- salvar deltas em formato binario/compacto;
- salvar estado real do otimizador;
- adicionar checksums;
- treinar e avaliar loss com `drm_transformer`;
- trocar diferenca finita por autograd quando PyTorch entrar.
