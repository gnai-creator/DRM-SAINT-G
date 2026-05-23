# Fase 7 - Runtime SAINT-G

Status: **concluida**.

Esta fase cria o primeiro runtime unificado para executar experimentos pequenos
de ponta a ponta com config, estimativa de memoria, treino, logs e checkpoint.

## Objetivo

Transformar os experimentos das fases anteriores em um fluxo reproduzivel:

```text
config -> inspect -> estimate -> train -> checkpoint -> resume -> merge
```

## Modulos Implementados

```text
saint/config/
saint/memory/
saint/checkpoints/
saint/adapters/
saint/runtime/
saint/cli.py
```

## CLI Inicial

Executar com:

```bash
python -m saint.cli inspect --config configs/runtime_smoke.json
python -m saint.cli estimate --config configs/runtime_smoke.json
python -m saint.cli train --config configs/runtime_smoke.json
python -m saint.cli resume --run runs/runtime_smoke
python -m saint.cli merge --run runs/runtime_smoke
```

## Config Inicial

```text
configs/runtime_smoke.json
```

Ela executa `mini_SAINT-G_dynamic_delta` no mini-transformer com mapa
`gradient_norm`.

## Artefatos do Run

O comando `train` grava:

```text
config.json
metrics.json
checkpoint.json
logs.jsonl
```

O comando `merge` grava:

```text
merged.json
```

## Criterio de Conclusao

A fase conclui quando o runtime:

- carrega config JSON;
- inspeciona o modelo;
- estima memoria;
- executa treino pequeno;
- grava logs e checkpoint;
- faz resume do checkpoint;
- produz artefato de merge;
- possui testes automatizados.

## Smoke Test

Comandos executados:

```bash
python -m saint.cli inspect --config configs/runtime_smoke.json
python -m saint.cli estimate --config configs/runtime_smoke.json --vram-gb 12
python -m saint.cli train --config configs/runtime_smoke.json
python -m saint.cli resume --run runs/runtime_smoke
python -m saint.cli merge --run runs/runtime_smoke
```

Resultado do treino:

```text
method: mini_SAINT-G_dynamic_delta
parameter_count: 30
test_loss: 0.00016531
train_loss: 0.00009658
fits_budget: true
estimated_bytes: 288
```

Artefatos gerados:

```text
runs/runtime_smoke/config.json
runs/runtime_smoke/metrics.json
runs/runtime_smoke/checkpoint.json
runs/runtime_smoke/logs.jsonl
runs/runtime_smoke/merged.json
```

Veredito:

```text
Fase 7 concluida pelo criterio inicial.
O runtime executa um experimento pequeno de ponta a ponta com logs e checkpoints.
```

## Proximos Passos

1. Avancar para Fase 8 - Checkpoint e Reconstituicao.
2. Expandir checkpoints para salvar deltas reais, nao apenas metricas.
3. Adicionar comandos equivalentes ao entrypoint `SAINT-G` quando houver pacote.
4. Integrar runtime com adaptador `drm_transformer`.
5. Trocar diferenca finita por autograd quando PyTorch entrar.
