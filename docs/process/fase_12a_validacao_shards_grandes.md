# Fase 12A - Validacao de Shards Grandes

Status: **concluida**.

## Objetivo

Validar se o formato shardado do checkpoint SAINT continua correto quando o
payload cresce e quando uma matriz individual precisa ser dividida em partes.

## Implementado

- sharding por matriz;
- sharding interno por faixas de linhas para matriz maior que `shard_bytes`;
- metadados `matrix_parts` por shard;
- remontagem de matriz dividida em multiplos shards;
- validacao SHA-256 por shard;
- benchmark sintetico de payload grande;
- medicao de tempo de escrita;
- medicao de tempo de leitura;
- medicao de pico de memoria durante leitura;
- testes de corrupcao de shard.

## Benchmark

Configuracao:

```text
matrix_count: 8
rows: 256
cols: 256
dtype: float16
shard_bytes: 65536
value_count: 524288
```

Resultado:

```text
format: saint_matrix_shards
shard_count: 16
payload_bytes: 1052000
write_elapsed_s: 0.1800
read_elapsed_s: 0.2984
read_peak_bytes: 17515952
checksum_validated: true
max_abs_error: 0.00000377
```

## Veredito

```text
Fase 12A concluida.
```

O checkpoint shardado agora suporta payloads maiores e matrizes individuais que
precisam ser divididas internamente, preservando validacao de integridade por
shard e reconstruindo a matriz original.

## Proximas Subfases

- Fase 12B - Merge Parcial;
- Fase 12C - Custo de I/O por Dtype;
- Fase 12D - Compatibilidade e Migracao;
- Fase 12E - Qualidade Numerica.
