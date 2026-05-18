# Fase 5 - Mini-Transformer

Status: **concluida**.

Esta fase valida SAINT em um modelo com matrizes acopladas por uma loss global,
antes de ir para um Transformer real maior.

## Objetivo

Verificar se a vantagem vista na Fase 4 em deltas repetidos ainda aparece quando
varias matrizes influenciam a mesma loss.

## Modelo Inicial

Foi criado um mini-transformer dependency-free e pequeno o suficiente para
testes unitarios:

```text
vocab_size: 8
d_model: 4
seq_len: 4
```

Matrizes:

```text
embed
w_q
w_k
w_v
w_o
w_mlp1
w_mlp2
w_head
```

O forward usa:

```text
embedding -> self-attention de ultima posicao -> residual -> MLP -> head
```

## Tarefa

A tarefa ainda e sintetica:

```text
modelo base congelado
modelo teacher = base + delta_target
student aprende deltas para aproximar logits do teacher
```

A loss e global:

```text
MSE(logits_student, logits_teacher)
```

## Modos Implementados

- `mini_full_delta`: treina todos os valores de delta;
- `mini_budgeted_delta`: treina valores individuais por sensibilidade inicial;
- `mini_block_budgeted_delta`: treina blocos livres `2x2`;
- `mini_saint_dynamic_delta`: codebook SAINT por blocos com escala, bias e residual.

## Por Que Diferenca Finita?

Nesta fase inicial, o objetivo e validar o acoplamento conceitual sem depender
de PyTorch.

Por isso, o gradiente e calculado por diferenca finita:

```text
grad = (loss(param + eps) - loss(param - eps)) / (2 * eps)
```

Isso nao e eficiente para escala real, mas torna o experimento pequeno,
deterministico e facil de auditar.

## Arquivos

Codigo:

```text
saint/transformer/model.py
saint/transformer/training.py
saint/transformer/saint_adapter.py
saint/transformer/benchmark.py
scripts/benchmark_mini_transformer_phase5.py
tests/test_transformer_phase5.py
```

## Perguntas

- Loss global com atualizacao local funciona?
- SAINT aprende quando varias matrizes estao acopladas?
- A vantagem em deltas repetidos aparece de novo?
- `budgeted_full_delta` continua sendo oraculo forte?
- Blocos livres explicam a maior parte do ganho ou o codebook ajuda?

## Criterio de Conclusao

A Fase 5 deve ser concluida somente se:

- SAINT reduzir loss em relacao ao modelo base;
- SAINT vencer ou empatar LoRA/baseline eficiente quando ela for adicionada;
- SAINT vencer `mini_block_budgeted_delta` ou `mini_budgeted_delta` em pelo menos alguns regimes;
- o resultado se mantiver em `repeated` e nao colapsar em `dense`;
- o benchmark tiver criterio automatico de sucesso/falha.

## Benchmark Inicial

Configuracao:

```text
seeds: 31, 32
delta_modes: repeated, dense
steps: 8
parameter_budget: 48
```

Arquivos gerados:

```text
runs/phase5_mini_transformer_initial/mini_transformer_rows.json
runs/phase5_mini_transformer_initial/mini_transformer_summary.json
runs/phase5_mini_transformer_initial/mini_transformer.md
```

Resultado agregado:

| Metodo | Runs | Test Loss Medio | Params Medios | Ganho/Parametro |
|---|---:|---:|---:|---:|
| mini_saint_dynamic_delta | 4 | 0.00001998 | 48.0 | 0.0000000205 |
| mini_full_delta | 4 | 0.00002063 | 160.0 | 0.0000000021 |
| mini_budgeted_delta_for_saint | 4 | 0.00002063 | 48.0 | 0.0000000070 |
| mini_block_budgeted_delta_for_saint | 4 | 0.00002064 | 48.0 | 0.0000000068 |

Leitura:

```text
SAINT venceu os controles neste primeiro teste pequeno,
mas a loss inicial ja e muito baixa.
```

Portanto:

```text
resultado positivo, mas ainda fraco;
nao conclui a Fase 5.
```

## Benchmark com Tarefa Mais Dificil

A dificuldade foi aumentada com:

```text
delta_scale: 3.0
```

Tambem foram adicionados:

- LoRA nas matrizes `w_q`, `w_v`, `w_o` e `w_head`;
- `mini_saint_per_matrix_delta`, com codebook separado por matriz;
- `mini_saint_dynamic_delta`, com codebook global compartilhado entre matrizes;
- criterio automatico de fechamento da Fase 5.

Arquivos gerados:

```text
runs/phase5_mini_transformer_harder_v2/mini_transformer_rows.json
runs/phase5_mini_transformer_harder_v2/mini_transformer_summary.json
runs/phase5_mini_transformer_harder_v2/mini_transformer_decision.json
runs/phase5_mini_transformer_harder_v2/mini_transformer.md
```

Resultado agregado:

| Metodo | Runs | Test Loss Medio | Params Medios | Ganho/Parametro |
|---|---:|---:|---:|---:|
| mini_saint_per_matrix_delta | 4 | 0.00019998 | 72.0 | 0.0000001269 |
| mini_saint_dynamic_delta | 4 | 0.00020029 | 48.0 | 0.0000001838 |
| mini_full_delta | 4 | 0.00020591 | 160.0 | 0.0000000201 |
| mini_budgeted_delta_for_saint | 4 | 0.00020591 | 48.0 | 0.0000000667 |
| mini_block_budgeted_delta_for_saint | 4 | 0.00020598 | 48.0 | 0.0000000653 |
| mini_lora_rank_1 | 4 | 0.00020912 | 36.0 | 0.0000000000 |
| mini_lora_rank_2 | 4 | 0.00020912 | 72.0 | 0.0000000000 |

Decisao automatica:

```text
passou: sim
regimes: 4
vence LoRA rank 1: 4/4
vence LoRA rank 2: 4/4
vence block_budgeted_delta: 4/4
vence budgeted_delta: 4/4
consolidacao global eficiente contra SAINT por matriz: 4/4
```

Leitura:

```text
SAINT global ficou ligeiramente pior que SAINT por matriz em loss media,
mas usou 48 parametros contra 72 e teve melhor ganho por parametro.

Isso suporta a ideia de consolidacao: compartilhar codebook entre matrizes pode
ser mais eficiente que manter codebooks separados.
```

Veredito:

```text
Fase 5 concluida pelo criterio inicial.
O resultado ainda e pequeno e sintetico, mas valida loss global com matrizes acopladas.
```

## Proximos Passos

1. Avancar para Fase 6 - Mapa de Sensibilidade.
2. Substituir diferenca finita por autograd quando entrar PyTorch.
3. Testar mini-transformer com dimensoes maiores.
4. Levar `mini_saint_dynamic_delta` como baseline inicial.
5. Manter LoRA, budgeted e block-budgeted como controles.
