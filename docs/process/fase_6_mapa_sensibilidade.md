# Fase 6 - Mapa de Sensibilidade

Status: **concluida**.

Esta fase mede quais heuristicas escolhem melhor os parametros ou blocos que
devem receber delta treinavel.

## Objetivo

Escolher regioes treinaveis melhor que uma selecao aleatoria, usando o
mini-transformer da Fase 5 como ambiente de teste.

## Metodos Implementados

- `random`: selecao aleatoria deterministica;
- `gradient_norm`: norma absoluta do gradiente inicial;
- `gradient_weight`: `abs(gradiente * peso_base)`;
- `mask_impact`: ganho de loss ao aplicar o delta alvo naquela coordenada;
- `fisher`: aproximacao diagonal `gradiente^2`;
- `activation_magnitude`: magnitude do peso base como proxy;
- `layer_error`: erro total da matriz/camada;
- `gain_per_byte`: ganho direto por parametro;
- `pattern_frequency`: frequencia de padrao de bloco ponderada por delta.

## Experimento

Cada metodo escolhe o mesmo orcamento:

```text
parameter_budget: 48
```

Depois, treina somente as coordenadas escolhidas com a mesma loss global do
mini-transformer.

## Arquivos

```text
saint/sensitivity/transformer.py
scripts/benchmark_sensitivity_phase6.py
tests/test_sensitivity_phase6.py
```

## Criterio Inicial

A fase avanca se pelo menos 3 metodos vencerem `random` em loss media.

Isso ainda nao conclui a fase inteira; e apenas o primeiro criterio para provar
que o mapa de sensibilidade carrega sinal util.

## Benchmark Inicial

Configuracao:

```text
seeds: 41, 42
delta_modes: repeated, dense
steps: 8
parameter_budget: 48
delta_scale: 3.0
```

Arquivos gerados:

```text
runs/phase6_sensitivity_initial/sensitivity_rows.json
runs/phase6_sensitivity_initial/sensitivity_summary.json
runs/phase6_sensitivity_initial/sensitivity_decision.json
runs/phase6_sensitivity_initial/sensitivity.md
```

Resultado agregado:

| Metodo | Runs | Test Loss Medio | Ganho/Parametro |
|---|---:|---:|---:|
| sensitivity_fisher | 4 | 0.00029264 | 0.0000000493 |
| sensitivity_gradient_norm | 4 | 0.00029264 | 0.0000000493 |
| sensitivity_gradient_weight | 4 | 0.00029269 | 0.0000000482 |
| sensitivity_gain_per_byte | 4 | 0.00029280 | 0.0000000458 |
| sensitivity_mask_impact | 4 | 0.00029280 | 0.0000000458 |
| sensitivity_layer_error | 4 | 0.00029347 | 0.0000000319 |
| sensitivity_pattern_frequency | 4 | 0.00029426 | 0.0000000155 |
| sensitivity_random | 4 | 0.00029433 | 0.0000000139 |
| sensitivity_activation_magnitude | 4 | 0.00029443 | 0.0000000118 |

Decisao:

```text
passou: sim
metodos que venceram random: 7
```

Leitura:

```text
O mapa de sensibilidade carrega sinal util.
Gradient norm e Fisher empataram como melhores metodos iniciais.
Magnitude do peso base nao foi um bom proxy neste experimento.
```

## Benchmark Final

Foram adicionados:

- resumo separado por `repeated` e `dense`;
- comparacao de coordenadas contra blocos `2x2`;
- `mini_SAINT-G_gradient_norm`, usando mapa de sensibilidade para alimentar SAINT-G;
- `sensitivity_accumulated_gradient`, acumulando gradientes durante warmup;
- criterio final automatico.

Arquivos gerados:

```text
runs/phase6_sensitivity_final/sensitivity_rows.json
runs/phase6_sensitivity_final/sensitivity_summary.json
runs/phase6_sensitivity_final/sensitivity_by_regime.json
runs/phase6_sensitivity_final/sensitivity_decision.json
runs/phase6_sensitivity_final/sensitivity_final_decision.json
runs/phase6_sensitivity_final/sensitivity.md
```

Resultado agregado:

| Metodo | Runs | Test Loss Medio | Ganho/Parametro |
|---|---:|---:|---:|
| mini_SAINT-G_default_for_sensitivity | 4 | 0.00029131 | 0.0000000769 |
| mini_SAINT-G_gradient_norm | 4 | 0.00029131 | 0.0000000769 |
| sensitivity_accumulated_gradient | 4 | 0.00029264 | 0.0000000493 |
| sensitivity_fisher | 4 | 0.00029264 | 0.0000000493 |
| sensitivity_gradient_norm | 4 | 0.00029264 | 0.0000000493 |
| block_sensitivity_gradient_norm | 4 | 0.00029268 | 0.0000000483 |
| sensitivity_random | 4 | 0.00029433 | 0.0000000139 |

Resultado por regime:

```text
dense: gradient_norm/fisher/acumulado venceram random.
repeated: gradient_norm/fisher/acumulado venceram random.
```

Decisao final:

```text
passou: sim
regimes aprovados: 2
melhor bloco 2x2 venceu random: sim
sensibilidade acumulada venceu random: sim
SAINT-G alimentado por sensibilidade empatou/venceu SAINT-G padrao: sim
```

Leitura:

```text
Gradient norm e Fisher continuam sendo os melhores mapas simples.
Blocos 2x2 preservam boa parte do sinal, mas perdem levemente para coordenadas.
Sensibilidade acumulada empatou com gradient_norm neste experimento pequeno.
mini_SAINT-G_gradient_norm empatou com o SAINT-G padrao porque o SAINT-G padrao ja usava
gradiente inicial como sinal de roteamento.
```

Veredito:

```text
Fase 6 concluida pelo criterio atual.
O proximo passo e usar esses mapas em um runtime/adapter mais realista.
```

## Proximos Passos

1. Avancar para Fase 7 - Runtime SAINT-G.
2. Usar `gradient_norm`/`fisher` como mapas iniciais.
3. Manter `random` como controle obrigatorio.
4. Testar mapas acumulados quando houver autograd.
5. Integrar mapas de sensibilidade ao roteador SAINT-G.
