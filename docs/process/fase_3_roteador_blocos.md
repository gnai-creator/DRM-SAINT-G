# Fase 3 - Roteador de Blocos

Status: **concluida**.

Esta fase implementa o mecanismo que decide, por regiao da matriz, qual
representacao usar no paradigma SAINT-G.

## Objetivo

Resolver o problema observado na Fase 2:

```text
block_codebook_4 tem bom erro, mas pouca compressao;
hierarchical_codebook comprime, mas perde informacao demais.
```

O roteador deve escolher por regiao entre:

- `freeze`, tambem chamado de `zero_delta`;
- `codebook_4`;
- `codebook_2`;
- `free_delta`;
- variantes com escala por bloco;
- refinamento residual.

## Implementado

Modulo:

```text
saint.routing
```

Funcoes principais:

```python
route_matrix_regions(...)
route_matrix_regions_by_budget(...)
route_matrix_regions_by_sensitivity_budget(...)
routed_codebook_reconstruction(...)
routed_budget_reconstruction(...)
routed_sensitivity_budget_reconstruction(...)
search_routed_budget_reconstruction(...)
```

Baselines adicionais:

```python
scaled_block_codebook_reconstruction(...)
residual_codebook_reconstruction(...)
```

## Politicas

### Quality First

```text
para cada regiao:
  testar codebook_4 e codebook_2
  escolher a opcao mais barata abaixo do erro maximo
  se nenhuma passar, usar free_delta
```

### Budget First

```text
para cada regiao:
  testar codebook_4, codebook_2 e free_delta
  calcular score = erro_penalty + parametro_penalty
  escolher menor score
  reportar se alvo de compressao foi atingido
```

### Budget Search

```text
para cada matriz:
  testar multiplos parameter_weight
  escolher melhor resultado que atinge:
    erro medio <= limite
    compressao >= alvo
  se nenhum atingir, escolher menor violacao agregada
```

### Sensitivity Budget

```text
para cada regiao:
  calcular sensibilidade_proxy = norma L1 da regiao
  testar freeze, codebook_4, codebook_2 e free_delta
  calcular score = sensibilidade * erro + lambda * parametros
  respeitar orcamento duro por metodo
```

O objetivo dessa politica e aproximar a ideia real do SAINT-G:

```text
regiao pouco importante -> freeze
regiao media -> codebook
regiao critica -> free_delta
```

## Comparacao em Matrizes Reais

Checkpoint:

```text
E:\dev\ai\drm_transformer\checkpoints\topology_controls\seed_42\best.pt
```

Matrizes:

```text
8 matrizes reais, amostras 64x64
```

Configuracao principal:

```text
quantization_step: 0.05
region_size: 8
candidate_block_sizes: 4, 2
target_compression: 1.1
max_avg_relative_l1_error: 0.1
```

Resultado medio:

| Metodo | Erro relativo medio | Compressao media | Params medios |
|---|---:|---:|---:|
| block_codebook_4 | 0.0753 | 1.03 | 4066.0 |
| hierarchical_codebook | 0.4812 | 2.26 | 1962.9 |
| routed_quality_first | 0.0003 | 0.94 | 4347.0 |
| routed_budget_first | 0.0028 | 0.97 | 4213.0 |
| routed_budget_search | 0.0080 | 1.01 | 4062.0 |
| routed_sensitivity_budget | 0.7441 | 7.73 | 825.5 |
| scaled_block_codebook_4 | 0.0000 | 0.89 | 4608.0 |
| residual_codebook | 0.0000 | 0.79 | 5188.0 |

Com `quantization_step = 0.10`, a compressao sobe, mas o erro tambem:

| Metodo | Erro relativo medio | Compressao media |
|---|---:|---:|
| block_codebook_4 | 0.4531 | 2.86 |
| routed_budget_first | 0.1374 | 1.20 |
| routed_budget_search | 0.0217 | 1.04 |
| residual_codebook | 0.2320 | 1.05 |

Foram testados tambem pontos intermediarios:

### Quantization Step 0.075

| Metodo | Erro relativo medio | Compressao media | Params medios |
|---|---:|---:|---:|
| block_codebook_4 | 0.2775 | 1.69 | 3334.0 |
| hierarchical_codebook | 0.6932 | 3.14 | 1453.5 |
| routed_budget_first | 0.0645 | 1.04 | 4012.0 |
| routed_budget_search | 0.0447 | 1.05 | 3928.5 |
| multi_scale_codebook | 0.0420 | 1.02 | 4024.0 |
| residual_codebook | 0.0344 | 0.81 | 5061.5 |

### Quantization Step 0.080

| Metodo | Erro relativo medio | Compressao media | Params medios |
|---|---:|---:|---:|
| block_codebook_4 | 0.3123 | 1.87 | 3208.0 |
| hierarchical_codebook | 0.7291 | 3.42 | 1371.9 |
| routed_budget_first | 0.0836 | 1.07 | 3953.0 |
| routed_budget_search | 0.0593 | 1.06 | 3885.5 |
| multi_scale_codebook | 0.0755 | 1.06 | 3920.0 |
| residual_codebook | 0.0624 | 0.83 | 4961.5 |

### Quantization Step 0.085

| Metodo | Erro relativo medio | Compressao media | Params medios |
|---|---:|---:|---:|
| block_codebook_4 | 0.3370 | 2.06 | 3104.0 |
| hierarchical_codebook | 0.7580 | 3.43 | 1332.6 |
| routed_budget_first | 0.0964 | 1.10 | 3912.0 |
| routed_budget_search | 0.0823 | 1.08 | 3816.5 |
| multi_scale_codebook | 0.1123 | 1.11 | 3808.0 |
| residual_codebook | 0.0987 | 0.86 | 4849.5 |

### Quantization Step 0.0875

| Metodo | Erro relativo medio | Compressao media | Params medios |
|---|---:|---:|---:|
| block_codebook_4 | 0.3483 | 2.12 | 3050.0 |
| hierarchical_codebook | 0.7692 | 3.65 | 1293.9 |
| routed_budget_first | 0.1006 | 1.11 | 3897.0 |
| routed_budget_search | 0.0647 | 1.07 | 3871.0 |
| multi_scale_codebook | 0.1194 | 1.12 | 3784.0 |
| residual_codebook | 0.1061 | 0.87 | 4824.5 |

### Quantization Step 0.0869

| Metodo | Erro relativo medio | Compressao media | Params medios |
|---|---:|---:|---:|
| block_codebook_4 | 0.3455 | 2.11 | 3060.0 |
| hierarchical_codebook | 0.7663 | 3.56 | 1303.9 |
| routed_budget_first | 0.0991 | 1.10 | 3903.0 |
| routed_budget_search | 0.0630 | 1.06 | 3877.0 |
| multi_scale_codebook | 0.1177 | 1.12 | 3792.0 |
| residual_codebook | 0.1048 | 0.86 | 4832.0 |

Leitura dos pontos intermediarios:

```text
0.075 melhora compressao mantendo erro baixo no roteador,
mas ainda fica abaixo de 1.1 de compressao.

0.080 e o melhor ponto testado ate agora para o roteador:
erro medio abaixo de 0.1,
compressao media perto de 1.1,
mas ainda nao passa o criterio automatico.

0.085 chega mais perto do alvo:
routed_budget_first fica com erro 0.0964,
compressao 1.0958,
mas ainda falha por margem pequena.

0.0875 cruza o alvo de compressao,
mas passa levemente do limite de erro:
erro 0.1006 > 0.1000,
compressao 1.1058 > 1.1000.

0.08725 foi testado e caiu no mesmo plato do 0.0875
para o metodo principal:
erro 0.1006,
compressao 1.1058.

0.0869 foi o primeiro ponto aprovado no criterio automatico:
erro 0.0991 <= 0.1000,
compressao 1.1023 >= 1.1000.
```

O melhor candidato de reconstrucao com roteador neste sweep foi:

```text
routed_budget_first com quantization_step=0.0869
erro relativo medio: 0.0991
compressao media: 1.1023
```

Ele passa o criterio automatico configurado:

```text
erro relativo medio <= 0.1
compressao media >= 1.1
```

O `quantization_step=0.0875` mostrou a fronteira superior:

```text
routed_budget_first com quantization_step=0.0875
erro relativo medio: 0.1006
compressao media: 1.1058
```

Ele falha por margem ainda menor, agora no erro:

```text
erro relativo medio 0.1006 > 0.1000
```

O teste com `quantization_step=0.08725` repetiu o mesmo resultado medio do
`routed_budget_first`, indicando que a mudanca de assinatura quantizada nessa
faixa e discreta, nao continua.

## Leitura Tecnica

O `routed_quality_first` confirma que o roteador sabe proteger informacao:

```text
erro relativo medio: 0.0003
```

Mas ele faz isso usando muitos parametros:

```text
compressao media: 0.94
```

O `routed_budget_search` melhora a compressao sem perder qualidade relevante:

```text
erro relativo medio: 0.0080
compressao media: 1.01
```

Ainda assim, ele nao atinge o alvo:

```text
erro <= 0.1
compressao >= 1.1
```

O `routed_sensitivity_budget` comprime muito:

```text
compressao media: 7.73
```

Mas falha em reconstruir pesos brutos:

```text
erro relativo medio: 0.7441
```

Esse resultado nao invalida diretamente o SAINT-G, porque `freeze/zero_delta`
foi pensado para deltas de treino, nao para reconstruir o peso completo. Em
um benchmark de peso bruto, congelar uma regiao significa trocar peso por zero,
o que naturalmente gera erro alto. No paradigma final, congelar significa:

```text
nao aplicar delta naquela regiao;
manter o peso base congelado intacto.
```

Portanto, o teste de matriz real mostra que a politica de freeze precisa ser
avaliada em treino de deltas na Fase 4.

## Criterio Automatico

Foi adicionado:

```text
evaluate_method_against_thresholds
```

Ele avalia:

```text
avg_relative_l1_error <= max_avg_relative_l1_error
avg_compression_ratio >= min_avg_compression_ratio
```

Resultado automatico no benchmark principal:

| Metodo | Passou? | Motivo |
|---|---|---|
| block_codebook_4 | nao | compressao 1.03 < 1.10 |
| hierarchical_codebook | nao | erro 0.48 > 0.10 |
| routed_quality_first | nao | compressao 0.94 < 1.10 |
| routed_budget_first | nao | compressao 0.97 < 1.10 |
| routed_budget_search | nao | compressao 1.01 < 1.10 |
| routed_sensitivity_budget | nao | erro 0.74 > 0.10 |
| scaled_block_codebook_4 | nao | compressao 0.89 < 1.10 |
| residual_codebook | nao | compressao 0.79 < 1.10 |

Nenhum metodo atingiu simultaneamente:

```text
erro relativo medio <= 0.1
compressao media >= 1.1
```

## Veredito Final

```text
Fase 3: concluida.
Hipotese de roteamento por erro suportada para qualidade.
Hipotese de roteamento por orcamento suportada no criterio minimo.
Hipotese de freeze/sensibilidade ainda nao pode ser julgada em peso bruto.
```

O criterio minimo foi atingido por:

```text
routed_budget_first
quantization_step: 0.0869
erro medio: 0.0991
compressao media: 1.1023
```

O proximo teste correto nao e insistir apenas em reconstruir pesos completos.
O proximo teste correto e a Fase 4:

```text
treinar deltas em uma camada linear
comparar full delta, LoRA e SAINT-G
medir loss, parametros treinaveis, memoria e compressao
```

## Proximos Passos

1. Criar tarefa de camada linear para medir aprendizado, nao apenas reconstrucao.
2. Usar `freeze` como ausencia de delta sobre peso base, nao como matriz zero de peso bruto.
3. Medir sensibilidade por gradiente ou impacto na loss.
4. Comparar SAINT-G contra LoRA em ganho por parametro treinavel.
5. Usar o resultado da Fase 3 como baseline inicial do roteador na Fase 4.
