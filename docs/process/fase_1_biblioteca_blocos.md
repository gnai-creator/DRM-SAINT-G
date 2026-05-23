# Fase 1 - Biblioteca de Blocos

Status: **concluida**.

Esta fase cria a base matematica inicial do SAINT-G para particionar, reconstruir, assinar, agrupar e analisar matrizes em blocos.

## Objetivo

Validar a primeira unidade do paradigma:

```text
W -> blocos -> grupos -> metricas -> W_recon
```

Antes de treinar uma LLM, SAINT-G precisa manipular matrizes de forma confiavel.

## Modulos

```text
SAINT-G.blocks.partition
SAINT-G.blocks.signatures
SAINT-G.blocks.grouping
SAINT-G.blocks.metrics
SAINT-G.blocks.codebook
```

## API Principal

```python
from SAINT-G.blocks import (
    analyze_block_reuse,
    build_fixed_codebook,
    group_blocks_by_signature,
    partition_matrix,
    reconstruct_matrix,
)

matrix = [
    [1, 2, 1, 2],
    [3, 4, 3, 4],
]

blocks = partition_matrix(matrix, block_size=(2, 2))
groups = group_blocks_by_signature(blocks, mode="exact")
codebook = build_fixed_codebook(blocks, mode="exact")
analysis = analyze_block_reuse(matrix, block_size=(2, 2))
reconstructed = reconstruct_matrix(blocks, original_shape=(2, 4))
```

## Funcionalidades

- particionamento de matriz 2D;
- reconstrucao com remocao de padding;
- suporte a blocos `2x2`, `3x3`, `4x4`, `5x5`, `6x6`, `8x8`, `16x16`;
- suporte a matrizes retangulares;
- validacao de matriz retangular;
- assinaturas exatas;
- assinaturas quantizadas;
- assinaturas estatisticas;
- agrupamento de blocos por assinatura;
- metricas de erro de reconstrucao;
- metricas de reutilizacao;
- codebook fixo inicial.

## Assinaturas

Modos disponiveis:

```text
exact
quantized
stats
```

`exact` agrupa blocos identicos.

`quantized` agrupa blocos parecidos em uma grade numerica.

`stats` usa estatisticas estruturais como soma, norma aproximada, traco e determinante para blocos `2x2`.

## Metricas

Metricas de reconstrucao:

- erro L1;
- erro L2;
- erro L1 relativo;
- erro maximo absoluto.

Metricas de reutilizacao:

- numero de blocos;
- numero de prototipos;
- numero de blocos repetidos;
- taxa de reutilizacao;
- valores originais estimados;
- valores de prototipos estimados;
- taxa de compressao estimada.

## Codebook Fixo

O codebook fixo usa o primeiro bloco de cada grupo como prototipo.

Ele ainda nao e treinavel.

Objetivo:

```text
validar ids, assignments e reutilizacao antes de implementar codebook aprendivel
```

## Testes

Rodar:

```bash
python -m unittest discover -s tests
```

Cobertura atual:

- round-trip em shape par;
- round-trip com padding;
- matrizes retangulares;
- varios tamanhos de bloco;
- agrupamento exato;
- agrupamento quantizado;
- assinatura estatistica com determinante `2x2`;
- metricas de reconstrucao;
- metricas de reutilizacao;
- codebook fixo.

## Criterio de Conclusao

A fase e considerada concluida porque SAINT-G ja consegue:

```text
particionar matriz
reconstruir matriz
detectar blocos iguais
detectar blocos parecidos por quantizacao
medir erro de reconstrucao
medir reutilizacao
criar codebook fixo
```

## Proxima Fase

Fase 2 - Benchmark de Reconstrucao.

Objetivo:

```text
comparar codebook multi-escala contra SVD, LoRA equivalente,
quantizacao e blocos fixos.
```
