# Fase 12B - Merge Parcial

Status: **concluida**.

## Objetivo

Permitir carregar e fundir apenas um subconjunto de matrizes de um checkpoint
SAINT-G shardado, evitando materializar todo o payload quando o alvo for parcial.

## Implementado

- leitura parcial por `matrix_names`;
- filtro de shards por metadados `matrix_parts`;
- leitura parcial dentro de um payload binario;
- `require_delta_payload(..., matrix_names=...)`;
- `merge_runtime(..., matrix_names=...)`;
- CLI `SAINT-G merge --matrix <nome>`;
- validacao de erro para matriz inexistente;
- testes para leitura parcial e merge parcial.

## Benchmark

Configuracao:

```text
matrix_count: 8
selected_count: 2
rows: 256
cols: 256
dtype: float16
shard_bytes: 65536
shard_count: 16
```

Resultado:

```text
full_read_elapsed_s: 0.2910
full_read_peak_bytes: 17524302
partial_read_elapsed_s: 0.0787
partial_read_peak_bytes: 4579606
partial_matrix_count: 2
selected_value_count: 131072
max_abs_error: 0.00000189
```

## Veredito

```text
Fase 12B concluida.
```

O runtime agora consegue fazer merge parcial carregando somente as matrizes
solicitadas. No benchmark sintetico, a leitura parcial reduziu pico de memoria
de aproximadamente 17.5 MB para 4.6 MB e tambem reduziu tempo de leitura.

## Proximas Subfases

- Fase 12C - Custo de I/O por Dtype;
- Fase 12D - Compatibilidade e Migracao;
- Fase 12E - Qualidade Numerica.
