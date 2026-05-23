# Fase 12C - Custo de I/O por Dtype

Status: **concluida**.

## Objetivo

Medir custo de checkpoint por dtype, comparando tamanho, tempo de escrita,
tempo de leitura, pico de memoria e erro numerico.

## Implementado

- benchmark `benchmark_dtype_io`;
- comparacao entre `float32`, `float16`, `bfloat16` e `int8`;
- razao de tamanho contra `float32`;
- erro maximo absoluto contra payload original;
- medicao de tempo de escrita;
- medicao de tempo de leitura;
- medicao de pico de memoria durante leitura;
- testes automatizados do sweep de dtype.

## Benchmark

Configuracao:

```text
matrix_count: 8
rows: 256
cols: 256
value_count: 524288
shard_bytes: 65536
```

Resultado:

| dtype | bytes | razao vs float32 | shards | escrita s | leitura s | pico leitura | erro max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| float32 | 2103968 | 1.0000 | 32 | 0.3124 | 0.3038 | 17504684 | 0.0000000005 |
| float16 | 1052000 | 0.5000 | 16 | 0.1771 | 0.3044 | 17521396 | 0.0000037720 |
| bfloat16 | 1052016 | 0.5000 | 16 | 0.1598 | 0.8330 | 17519142 | 0.0000602539 |
| int8 | 526121 | 0.2501 | 8 | 0.2240 | 0.2609 | 18066662 | 0.0000480315 |

## Leitura Tecnica

- `float16` oferece a melhor troca inicial: metade do tamanho de `float32` com
  erro baixo no payload sintetico.
- `bfloat16` tambem reduz tamanho pela metade, mas a implementacao Python pura
  atual le mais devagar.
- `int8` reduz para aproximadamente 25% do tamanho, mas deve ser tratado como
  formato agressivo ate validacao em tarefa real.
- O pico de memoria de leitura ainda e dominado pela materializacao em listas
  Python; reduzir isso depende das proximas fases de leitura parcial por trecho
  e integração com tensores.

## Veredito

```text
Fase 12C concluida.
```

SAINT-G agora mede custo de I/O por dtype e tem dados objetivos para escolher o
formato de checkpoint conforme tamanho, velocidade e erro numerico.

## Proximas Subfases

- Fase 12D - Compatibilidade e Migracao;
- Fase 12E - Qualidade Numerica.
