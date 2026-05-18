# Fase 10 - Checkpoint Robusto

Status: **concluida**.

Esta fase transforma checkpoints SAINT em artefatos compactos, versionados e
verificaveis.

## Objetivo

Separar metricas leves de payload pesado e permitir:

```text
treinar -> salvar checkpoint compacto -> validar -> retomar -> fundir -> avaliar
```

## Formato

O runtime grava:

```text
runs/<exp>/
  checkpoint.json
  metrics.json
  deltas.saintbin
  optimizer.saintopt
  logs.jsonl
```

## Manifesto

`checkpoint.json` e um manifesto leve:

```text
format: saint_checkpoint
format_version: 1
files:
  - path: deltas.saintbin
    payload: delta
    sha256: ...
  - path: optimizer.saintopt
    payload: optimizer_state
    sha256: ...
```

Observacao: a Fase 12D migrou o manifesto atual para `format_version: 2`,
mantendo leitura automatica de manifestos v1.

## Payload de Deltas

`deltas.saintbin` usa:

```text
magic: SAINTMAT1
header: JSON com shapes, offsets e dtype
payload: float32 little-endian
```

Isso reduz o tamanho contra JSON e preserva o suficiente para reconstruir o
modelo no `merge`.

## Estado do Otimizador

`optimizer.saintopt` usa:

```text
magic: SAINTOPT1
header: JSON
payload: zlib(JSON)
```

O formato aceita estado real de otimizador quando o metodo fornecer
`optimizer_state_payload`. Quando o metodo ainda nao exporta estado completo, o
runtime salva pelo menos o resumo versionado de `optimizer_state_values`.

## Validacao

`resume` e `merge` chamam validacao do bundle:

- verifica `format_version`;
- verifica SHA-256 de cada arquivo;
- carrega `deltas.saintbin` quando `has_delta_payload=true`;
- carrega `optimizer.saintopt`;
- rejeita payload corrompido antes de retomar ou fundir.

## Implementado

- `saint/checkpoints/robust.py`;
- manifesto robusto em `checkpoint.json`;
- deltas binarios em `deltas.saintbin`;
- estado de otimizador em `optimizer.saintopt`;
- `metrics.json` sem payload pesado;
- validacao de integridade no `resume`;
- validacao de integridade no `merge`;
- teste de corrupcao de payload.

## Veredito

```text
Fase 10 concluida para o runtime atual.
```

## Continuidade

Os itens de escalabilidade foram movidos para a Fase 11:

- estado completo de AdamW no caminho DRM autograd;
- suporte a shards;
- leitura por mmap;
- dtypes opcionais como float16, bfloat16 e int8;
- ponto de migracao entre versoes de checkpoint.
