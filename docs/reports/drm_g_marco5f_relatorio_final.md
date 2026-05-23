# Relatorio Final DRM-G Marco 5F

Status: **concluido com avanco condicional para Fase 16**.

Este relatorio fecha a fase DRM-G ate o Marco 5. O veredito e pragmatico:
SAINT-G deve avancar para a Fase 16, mas como evidencia parcial, nao como
vitoria absoluta contra `full_module_linear`.

## Resumo dos Marcos

| marco | resultado | leitura |
|---|---|---|
| 5A | artefato `.pt` consolidado salvo e reavaliado | consolidacao linear em disco funciona |
| 5B | 4/4 seeds com ganho positivo e retencao | nao houve regressao no split antigo medido |
| 5C | Phi competitivo na media, perdeu melhor caso | `A Phi B` segue viavel, mas nao domina full-module |
| 5D | multilingual 5M repetiu sinal de media/estabilidade | efeito nao ficou restrito ao DRM 3.5M |
| 5E | `partial_pass`, 7/8 eixos | criterio automatico reproduziu o veredito manual |

## Melhor Configuracao Atual

Para continuar a pesquisa, a melhor linha nao e o enxerto progressivo antigo. A
melhor familia atual e:

```text
A Phi B em modulo linear
target inicial: blocks.1.attn.out_proj
baseline de escala: configs/scaling/multilingual/5m.yaml
variantes prioritarias:
  phi_zero_full_rank
  phi_ls_residual_full_rank
  phi_ls_train_ab_half_rank
```

No DRM 3.5M, o melhor resultado medio foi:

| metodo | mean_gain | mean_gain/param | positivos |
|---|---:|---:|---:|
| `phi_zero_4096` | 0.017202 | 4.199748e-06 | 4 / 4 |
| `full_module_linear` | 0.016382 | 3.999550e-06 | 3 / 4 |

No DRM multilingual 5M, o melhor resultado medio foi:

| metodo | mean_gain | mean_gain/param | positivos |
|---|---:|---:|---:|
| `phi_ls_train_ab_half_rank` | 0.009759 | 8.471776e-07 | 3 / 4 |
| `full_module_linear` | 0.003337 | 3.621034e-07 | 2 / 4 |

## Metricas Principais

| metrica | resultado |
|---|---|
| validation loss antes/depois | 5A: `10.805441 -> 10.804946` |
| validation gain consolidado | 5A: `0.0004949569702148438` |
| ganho por parametro treinavel | 5C Phi medio: `4.199748e-06`; 5D Phi medio: `8.471776e-07` |
| checkpoint do enxerto consolidado | 5A `.pt`: `13,902,349` bytes |
| checksum do `.pt` | `68544e26197a4a83b1c3789cdd2dc92d599b745213f25e48ad8da41d73e8642e` |
| diferenca hook/saved | `0.0` |
| regressao em exemplos antigos | 5B: negativa em 4/4 seeds |
| taxa aprovacao/descarte | 5B melhor seed: `1 approved`, `1 rejected`, `1 deferred`; seed 33: `2 approved`, `0 rejected`, `1 deferred` |
| memoria CUDA por etapa | smoke CUDA anterior: routing `71,483,904` bytes, train `51,806,720` bytes, eval `45,419,520` bytes |
| tempo de treino | 5C/5D pequenos: subsegundo por variante neste regime CPU |

## Criterio Automatico

O Marco 5E gerou:

```text
status: partial_pass
passed: true
passed_axes: 7 / 6
```

| eixo | passou |
|---|---:|
| `artifact_reproducible` | true |
| `retention_win` | true |
| `best_case_win` | false |
| `mean_multiseed_win` | true |
| `stability_win` | true |
| `checkpoint_size_win` | true |
| `memory_win` | true |
| `compression_win` | true |

O eixo que falhou importa: `best_case_win=false`. Portanto, a fase nao deve ser
descrita como "SAINT-G vence full-module". A descricao correta e:

```text
SAINT-G/Phi e competitivo na media multiseed e mais estavel em alguns regimes,
mas ainda nao domina o melhor caso absoluto de full-module.
```

## Limites Conhecidos

- Os benchmarks ainda usam fixtures curtos e micro-regimes.
- O melhor caso individual ainda pertence a `full_module_linear`.
- `phi_ls_train_ab_half_rank` pode exceder o mesmo budget nominal do full-module.
- A memoria CUDA foi medida em smoke pequeno, nao em run longo no hardware alvo.
- Retencao foi validada para enxertos progressivos e precisa ser repetida para as
  variantes Phi mais fortes.
- Falta testar Phi multi-stage:

```text
Delta = A1 Phi1 B1 + A2 Phi2 B2
```

- Falta capar `train_ab` para budget equivalente a full-module.

## Recomendacao Tecnica

DRM-G deve avancar para Fase 16, mas com tres condicoes:

1. Usar `configs/scaling/multilingual/5m.yaml` como baseline imediato de entrada.
2. Carregar `phi_zero_full_rank` e `phi_ls_train_ab_half_rank` como baselines Phi.
3. Focar a Fase 16 em infraestrutura, escala e criterio multi-eixo, nao em afirmar
   superioridade absoluta contra full-module.

## Respostas de Fechamento

DRM-G deve avancar?

Sim, como `partial_pass`. A evidencia e suficiente para escalar o experimento,
mas nao para alegar dominancia total.

Qual baseline carregar para a proxima fase?

`drm_transformer/configs/scaling/multilingual/5m.yaml`.

O foco deve ser qualidade, escala ou infraestrutura?

Primeiro infraestrutura e escala controlada. Qualidade deve continuar sendo
medida, mas o risco principal antes da Fase 16 e operacional: memoria, checkpoint,
retencao, consolidacao e repetibilidade em modelos maiores.

## Proxima Fase

A Fase 16 deve iniciar com:

- baseline DRM multilingual 5M;
- Phi full-rank e half-rank;
- retencao das variantes Phi;
- CUDA real no hardware alvo;
- checkpoint recomponivel;
- comparacao multi-eixo contra full-module.

Se o 5M mantiver `partial_pass` com memoria controlada, o proximo salto deve ser
um DRM maior de scaling, nao 70B direto.
