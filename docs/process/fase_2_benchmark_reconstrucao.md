# Fase 2 - Benchmark de Reconstrucao

Status: **concluida**.

Esta fase testa se representacoes por codebook e blocos conseguem reconstruir matrizes com bons tradeoffs de erro, parametros, compressao e reutilizacao.

## Objetivo

Responder empiricamente:

```text
codebooks de blocos conseguem representar matrizes melhor
que baselines simples em algum regime?
```

## Implementado

Modulos:

```text
saint.reconstruction.generators
saint.reconstruction.baselines
saint.reconstruction.benchmark
saint.reconstruction.matrix_ops
```

Geradores:

- matriz gaussiana;
- matriz low-rank;
- matriz sparse;
- matriz com blocos repetidos.

Baselines iniciais:

- original sem compressao;
- quantizacao uniforme;
- low-rank por power iteration/deflation;
- block-codebook de tamanho unico;
- multi-scale codebook simples.

Runner:

```python
results = run_reconstruction_benchmark(cases, baselines)
```

Metricas:

- erro L1;
- erro L2;
- erro relativo;
- erro maximo absoluto;
- parametros estimados;
- compressao estimada;
- tempo;
- taxa de reutilizacao.

## Benchmark Inicial

Configuracao:

```text
matrizes: 8x8
casos: gaussian, low_rank, sparse, repeated_blocks
block sizes: 2 e 4
quantization step: 0.25
low_rank rank: 2
```

Resultados resumidos:

| Caso | Metodo | Erro relativo | Params | Compressao | Reuso |
|---|---:|---:|---:|---:|---:|
| gaussian_8x8 | original | 0.0000 | 64 | 1.00 | 0.00 |
| gaussian_8x8 | quantization | 0.0838 | 79 | 0.81 | 0.00 |
| gaussian_8x8 | low_rank_2 | 0.6627 | 34 | 1.88 | 0.00 |
| gaussian_8x8 | block_codebook_2 | 0.0000 | 80 | 0.80 | 0.00 |
| gaussian_8x8 | block_codebook_4 | 0.0000 | 68 | 0.94 | 0.00 |
| gaussian_8x8 | multi_scale | 0.0000 | 68 | 0.94 | 0.00 |
| low_rank_8x8 | low_rank_2 | 0.0000 | 34 | 1.88 | 0.00 |
| sparse_8x8 | block_codebook_2 | 0.0000 | 48 | 1.33 | 0.50 |
| sparse_8x8 | multi_scale | 0.0000 | 48 | 1.33 | 0.50 |
| repeated_blocks_8x8 | block_codebook_2 | 0.0000 | 28 | 2.29 | 0.81 |
| repeated_blocks_8x8 | multi_scale | 0.0000 | 28 | 2.29 | 0.81 |

## Leitura Inicial

O resultado inicial confirma algo esperado:

- matrizes gaussianas nao reutilizam padroes de bloco;
- matrizes low-rank favorecem baseline low-rank;
- matrizes sparse ja geram reuso moderado;
- matrizes com blocos repetidos favorecem codebook de blocos;
- multi-scale simples escolhe o melhor candidato entre tamanhos testados.

Isso ainda nao prova sucesso do paradigma, mas valida que o benchmark consegue diferenciar regimes.

## Benchmark em Matrizes Reais

Foi adicionado o script:

```text
scripts/benchmark_drm_matrices.py
```

Ele carrega um checkpoint real do `drm_transformer`, extrai matrizes 2D selecionadas e roda os metodos de reconstrucao do SAINT.

Checkpoint usado:

```text
E:\dev\ai\drm_transformer\checkpoints\topology_controls\seed_42\best.pt
```

Configuracao:

```text
matrizes: 8 matrizes reais
amostra por matriz: 64x64
quantization_step: 0.05
low_rank: 4
block sizes: 8, 4, 2
```

Metodos comparados:

- original;
- quantizacao uniforme;
- low_rank_4;
- block_codebook_2;
- block_codebook_4;
- multi_scale_codebook;
- hierarchical_codebook.

Resultados agregados sem contar `original`:

| Metodo | Erro relativo medio | Compressao media | Leitura |
|---|---:|---:|---|
| quantization_step_0.05 | 0.4995 | 1.00 | erro alto, sem compressao real |
| low_rank_4 | 0.6816 | 7.94 | comprime muito, mas erro alto |
| block_codebook_2 | 0.5103 | 2.37 | comprime, mas erro alto |
| block_codebook_4 | 0.0753 | 1.03 | melhor tradeoff atual em matrizes reais |
| multi_scale_codebook | 0.0000 | 0.98 | lossless, mas sem compressao |
| hierarchical_codebook | 0.4812 | 2.26 | comprime, mas erro alto |

Melhor por erro em todas as matrizes reais:

```text
multi_scale_codebook
```

Mas isso nao e uma vitoria forte, porque o metodo escolheu representacoes praticamente sem compressao:

```text
erro: 0.0000
compressao media: 0.98
```

Melhor tradeoff pratico atual nas matrizes reais:

```text
block_codebook_4
```

Porque apresentou:

```text
erro relativo medio: 0.0753
compressao media: 1.03
```

O `hierarchical_codebook` atual:

```text
erro relativo medio: 0.4812
compressao media: 2.26
```

Ou seja, comprime melhor, mas perde informacao demais.

## Onde SAINT Vence

SAINT vence claramente em matrizes sinteticas com estrutura local:

- matrizes sparse;
- matrizes com blocos repetidos;
- matrizes onde ha reutilizacao real de padroes.

Nesses regimes, block-codebook e multi-scale conseguem:

- erro zero ou baixo;
- compressao acima de 1;
- alta taxa de reutilizacao.

## Onde SAINT Perde

SAINT perde ou nao mostra vantagem clara em:

- matrizes gaussianas;
- matrizes low-rank, onde low-rank/SVD e mais natural;
- matrizes reais densas sem reuso obvio de blocos pequenos;
- modo hierarchical atual, que comprime mas gera erro alto.

## Veredito da Fase 2

Resultado:

```text
hipotese parcialmente suportada
```

A Fase 2 mostrou que:

- o benchmark e capaz de diferenciar regimes;
- codebooks de blocos funcionam bem quando ha estrutura local;
- em matrizes reais do `drm_transformer`, reuso exato/local nao e forte o suficiente com o metodo atual;
- `hierarchical_codebook` precisa de roteamento melhor para nao trocar qualidade por compressao agressiva demais;
- `block_codebook_4` e o melhor ponto de partida nas matrizes reais testadas.

Decisao:

```text
avancar para Fase 3 - Roteador de Blocos
```

Motivo:

```text
o gargalo agora e decisao por regiao.
O benchmark mostrou que tamanho fixo e hierarquia simples nao bastam.
A proxima hipotese precisa testar um roteador que escolha entre:
congelar, 8x8, 4x4, 2x2, delta livre ou LoRA auxiliar.
```

Avanco com restricao:

```text
SAINT nao deve assumir que codebook hierarquico atual ja e bom.
Fase 3 deve existir para melhorar selecao local e reduzir erro.
```

## Limitacoes Restantes

- O low-rank e uma aproximacao simples, nao SVD completa.
- Ainda nao ha codebook treinavel.
- A estimativa de parametros e simples.
- A estimativa de memoria ainda nao mede memoria real do processo.
- O benchmark real usou amostras `64x64`, nao matrizes inteiras grandes.
- O hierarchical atual usa regra simples de reuso, nao sensibilidade/loss.

## Proximos Passos

1. Implementar Fase 3 - Roteador de Blocos.
2. Usar erro de reconstrucao para escolher tamanho por regiao.
3. Comparar roteador contra `block_codebook_4` e `hierarchical_codebook`.
4. Melhorar baseline low-rank ou integrar SVD quando NumPy estiver disponivel.
5. Medir tempo e memoria de forma mais consistente.
6. Criar criterio automatico de sucesso/falha por benchmark.

## Criterio de Conclusao da Fase

Esta fase e considerada concluida porque agora ha:

- benchmark reproduzivel em matrizes sinteticas;
- benchmark em matrizes reais do `drm_transformer`;
- comparacao contra low-rank, quantizacao, codebook tamanho unico e hierarchical;
- relatorio com casos onde SAINT vence e perde;
- decisao clara de avancar para roteador de blocos.
