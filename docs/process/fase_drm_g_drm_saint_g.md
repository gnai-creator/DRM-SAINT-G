# Fase DRM-G - DRM-SAINT-G

Status: **em andamento**.

## Objetivo

Testar crescimento progressivo do `drm_transformer` por enxertos treinaveis.

A ideia central e evitar tentar nascer com um DRM grande demais para o hardware
disponivel. Em vez disso, o modelo cresce por ciclos pequenos, preservando um
nucleo congelado e treinando apenas enxertos compactos com DRM-SAINT-G-Phi.

## Hipotese

DRM-SAINT-G deve permitir:

```text
DRM pequeno estavel
  -> modulo novo anexado
  -> treino local por DRM-SAINT-G-Phi
  -> validacao global
  -> consolidacao progressiva
```

Se funcionar, o projeto ganha um caminho mais realista para sair de modelos
pequenos sem exigir pre-training full de um modelo gigante.

## Modelo Mental

O enxerto nao e apenas LoRA em uma matriz existente.

Ele e um modulo novo ou uma extensao estrutural do DRM. DRM-SAINT-G-Phi treina a ponte
entre o nucleo antigo e a capacidade nova:

```text
Delta W = A Phi B
```

onde:

- `A` projeta o estado do DRM para um espaco local;
- `Phi` e o operador relacional/geometrico treinavel;
- `B` projeta o resultado de volta para o fluxo do modelo.

## Entregas

- especificacao de `DRMGraftConfig`;
- registro de modulos enxertaveis do DRM;
- congelamento explicito do nucleo antigo;
- criacao de enxertos pequenos em camadas selecionadas;
- treino do enxerto com DRM-SAINT-G-Phi;
- checkpoint separado por enxerto;
- validacao antes/depois no mesmo corpus;
- criterio para consolidar ou descartar enxertos;
- comparacao contra adicionar parametros e treinar full no mesmo budget.

## Marcos

### Marco 1 - Enxerto Simulado

Status: **concluido como smoke inicial**.

Criar um experimento pequeno, dependency-free ou PyTorch simples, em que um DRM
reduzido recebe um modulo novo e treina apenas o enxerto.

Implementacao:

- criado adapter `saint.adapters.drm_grafting`;
- criado metodo runtime `drm_g_saint_phi_graft`;
- criado config `configs/drm_g_marco1_3_5m.json`;
- usado baseline canonico `drm_transformer/configs/baselines/small_3.5M.yaml`;
- nucleo DRM congelado;
- enxerto aplicado por hook em `final_norm`;
- atualizacao treinavel:

```text
hidden' = hidden + hidden A Phi B
```

Resultado do smoke:

| metrica | valor |
|---|---:|
| parametros do DRM base | 3.468.581 |
| parametros treinaveis do enxerto | 64 |
| phi_rank | 8 |
| base_loss | 10.825858 |
| graft_loss | 10.822562 |
| validation_gain | 0.003296 |
| validation_gain_per_parameter | 5.1498e-05 |
| dense_budget_loss | 10.793698 |
| dense_budget_gain_per_parameter | 5.0249e-04 |

Leitura:

DRM-SAINT-G ja executa um ciclo minimo de enxerto sobre a arquitetura 3.5M real:
carrega config canonica, congela o nucleo, treina apenas `Phi` e salva
checkpoint do runtime.

O resultado ainda nao vence a baseline densa com o mesmo numero de parametros.
Isso e esperado no primeiro smoke, porque `A` e `B` ainda sao projecoes
aleatorias fixas. O proximo marco deve tornar o enxerto mais realista antes de
consolidar:

- inicializar `A` e `B` por ativacao/gradiente;
- medir validacao em corpus real ou fixture tokenizada;
- salvar payload real do enxerto, nao apenas metricas;
- comparar contra treino direto de um modulo novo equivalente;
- testar enxerto em pontos internos do bloco, nao apenas `final_norm`.

### Marco 2 - Enxerto Real no DRM Transformer

Status: **concluido como smoke inicial**.

Integrar o mecanismo ao `drm_transformer`, congelando o nucleo e treinando um
enxerto em uma ou mais matrizes reais.

Mudancas:

- `A` e `B` podem ser inicializados por:
  - `random`;
  - `activation`;
  - `gradient`;
  - `activation_gradient`;
- o ponto de enxerto passou a ser configuravel por `target_module`;
- treino e validacao usam batches separados por `data_seed` e `validation_seed`;
- o Marco 2 usa o baseline real `small_3.5M.yaml`.

Config oficial:

```text
config: configs/drm_g_marco2_3_5m_gradient_block1.json
target_module: blocks.1
projection_init: gradient
phi_rank: 8
trainable_params: 64
learning_rate: 0.005
steps: 2
```

Resultado oficial:

| metrica | valor |
|---|---:|
| parametros do DRM base | 3.468.581 |
| parametros treinaveis | 64 |
| base_loss validacao | 10.835747 |
| graft_loss validacao | 10.834869 |
| validation_gain | 0.000877 |
| validation_gain_per_parameter | 1.3709e-05 |
| dense_budget_loss | 10.835972 |
| dense_budget_gain | -0.000225 |

Leitura:

O Marco 2 melhora o Marco 1 em realismo: o enxerto nao depende mais apenas de
projecoes aleatorias e a validacao ja usa outro batch. A melhor configuracao do
smoke foi `blocks.1` com inicializacao por gradiente. Nesse ponto,
DRM-SAINT-G melhorou a validacao enquanto a baseline densa de mesmo budget
piorou.

Varredura curta:

| alvo | inicializacao | validation_gain | leitura |
|---|---|---:|---|
| `blocks.0` | `activation` | 0.001176 | melhor ganho bruto, mas dense tambem melhorou |
| `blocks.1` | `gradient` | 0.000877 | melhor ponto contra dense no mesmo budget |
| `blocks.1` | `activation` | 0.000472 | positivo |
| `blocks.2` | `activation_gradient` | 0.000427 | positivo |
| `final_norm` | `random` | -0.000081 | quase neutro |
| `final_norm` | `activation_gradient` | -0.001027 | piorou |

Proximo passo:

- salvar payload real do enxerto (`A`, `Phi`, `B`) no checkpoint;
- aplicar merge/eval recompondo o enxerto;
- testar mais seeds e mais exemplos de validacao;
- comparar contra baseline densa em multiplos pontos internos;
- testar `blocks.0` com `activation` como candidato de maior ganho bruto.

### Marco 3 - Consolidacao

Status: **implementado como infraestrutura, qualidade ainda negativa em dados reais**.

Testar se o enxerto pode ser consolidado no checkpoint sem perder a melhoria de
validacao.

Mudancas implementadas:

- checkpoint agora salva payload real do enxerto em `graft.drm-g.json`;
- payload inclui `A`, `Phi`, `B`, `target_module`, `scale` e metadados;
- adicionado formato `drm_graft_payload_json`;
- adicionado metodo runtime `drm_g_saint_phi_eval`;
- eval recomposto carrega o payload do checkpoint e reaplica o enxerto via hook;
- `optimizer.saintopt` agora salva o `state_dict` real do AdamW do enxerto;
- dados reais tokenizados podem ser lidos de `drm_transformer/data/baseline`;
- checkpoints podem gravar `consolidation.drm-g.json`;
- alvos lineares compativeis geram `delta_weight` para merge no `state_dict`;
- alvos nao-lineares, como `blocks.1`, ficam marcados como `hook_required`;
- cada run calcula `graft_decision` com `approve` ou `reject`;
- adicionado sweep `scripts/benchmark_drm_g_marco3.py`;
- sweep testa seeds `31`, `32`, `33`, batch de validacao maior e pontos internos:
  - `blocks.0`;
  - `blocks.1`;
  - `blocks.2`;
  - `final_norm`.

Validacao de recomposicao:

```text
train checkpoint: runs/drm_g_marco2_3_5m_gradient_block1
eval config: configs/drm_g_marco3_eval_payload.json
payload: graft.drm-g.json
target_module: blocks.1
projection_init: gradient
```

Resultado recomposto:

| metrica | valor |
|---|---:|
| base_loss | 10.835747 |
| graft_loss recomposto | 10.834869 |
| validation_gain | 0.000877 |
| validation_gain_per_parameter | 1.3709e-05 |

O eval recomposto reproduz a loss do checkpoint treinado, portanto `A`, `Phi` e
`B` ja sao suficientes para reativar o enxerto.

Resultado do sweep:

| seed | alvo | inicializacao | validation_gain | dense_gain | vence dense |
|---:|---|---|---:|---:|---|
| 32 | `final_norm` | `activation` | 0.001096 | -0.000362 | sim |
| 32 | `blocks.1` | `gradient` | 0.000933 | -0.000139 | sim |
| 31 | `blocks.2` | `gradient` | 0.000813 | 0.000028 | sim |
| 31 | `final_norm` | `activation` | 0.000638 | -0.000343 | sim |
| 32 | `final_norm` | `gradient` | 0.000597 | -0.000362 | sim |

Leitura:

O Marco 3 confirma que o enxerto e recomponivel e que o melhor ponto muda com
seed/batch. `blocks.0 + activation`, que parecia forte no Marco 2, nao foi o
melhor no sweep multiseed; `final_norm + activation` e `blocks.1 + gradient`
ficaram mais fortes neste teste.

Pendencias para fechar Marco 3:

- validar que o `delta_weight` consolidado preserva exatamente a loss em alvos
  lineares;
- encontrar uma configuracao com ganho positivo em dados reais tokenizados;
- ampliar validacao real para mais exemplos;
- expandir a politica `graft_decision` para filas com multiplos enxertos.

Smoke com tokens reais:

| config | alvo | merge | validation_gain | decisao |
|---|---|---|---:|---|
| `drm_g_marco3_real_tokens.json` | `blocks.1` | hook required | -0.001091 | reject |
| `drm_g_marco3_consolidated_linear.json` | `blocks.1.attn.out_proj` | state delta | -0.000483 | reject |

Leitura:

As quatro pendencias tecnicas foram enderecadas no runtime, mas a qualidade
ainda nao passou em dados reais tokenizados. Isso e um resultado importante:
o Marco 3 agora consegue rejeitar automaticamente enxertos ruins em vez de
apenas salvar qualquer delta.

### Marco 4 - Crescimento Progressivo

Status: **passou no criterio minimo em smoke com tokens reais**.

Repetir o ciclo mais de uma vez:

```text
DRM-1 -> DRM-1+G1 -> DRM-1+G1+G2
```

O objetivo e medir se multiplos enxertos acumulam capacidade ou entram em
conflito.

Implementacao:

- novo metodo runtime `drm_g_saint_phi_progressive`;
- config `configs/drm_g_marco4_progressive_real_tokens.json`;
- payload sequencial `drm_graft_sequence_payload`;
- reaplicacao acumulada dos enxertos aprovados como hooks;
- rejeicao automatica por `graft_decision`;
- eval recomposto de sequencias via `drm_g_saint_phi_eval`;
- comparacao por candidato contra `DenseBudgetGraft` no mesmo ponto.

Resultado oficial:

| etapa | alvo | init | loss antes | loss depois | dense gain | decisao |
|---:|---|---|---:|---:|---:|---|
| 1 | `blocks.2` | `activation` | 10.792871 | 10.792534 | -0.000303 | approve |
| 2 | `final_norm` | `activation` | 10.792534 | 10.792057 | -0.000107 | approve |

Resumo:

| metrica | valor |
|---|---:|
| base_loss | 10.792871 |
| final_loss | 10.792057 |
| sequence_gain | 0.000814 |
| enxertos aprovados | 2 |
| parametros treinaveis | 128 |
| gain/param recomposto | 6.3628e-06 |

O eval recomposto do checkpoint sequencial reproduziu:

```text
base_loss: 10.792871
graft_loss: 10.792057
validation_gain: 0.000814
```

Leitura:

O Marco 4 demonstra o primeiro ciclo `G1 -> G2` em que o segundo enxerto melhora
a validacao sem destruir o ganho anterior. Os dois candidatos tambem venceram a
baseline densa local de mesmo budget neste smoke.

Pendencias:

- repetir com mais seeds;
- usar mais textos reais de treino/validacao;
- testar sequencias com alvos lineares consolidaveis;
- salvar politica de fila para aprovar, rejeitar ou adiar enxertos;
- medir conflito quando mais de dois enxertos sao acumulados.

## Metricas

- validation loss antes/depois;
- ganho por parametro treinavel;
- tamanho do checkpoint do enxerto;
- memoria CUDA por etapa;
- tempo de roteamento;
- tempo de treino;
- regressao em exemplos antigos;
- taxa de enxertos aprovados versus descartados.

## Criterio de Conclusao

A fase passa se pelo menos um ciclo completo demonstrar:

- melhoria de validacao contra base congelada;
- checkpoint recomponivel do enxerto;
- consolidacao sem regressao clara;
- ganho por parametro competitivo contra baseline full no mesmo budget;
- memoria controlada no hardware alvo.

## Relacao com Fase 16

Fase 16 deve escalar a estrategia que sair daqui.

Se DRM-SAINT-G funcionar, a escala 70B nao deve ser tratada apenas como adaptacao
de pesos existentes. Ela deve ser tratada como crescimento controlado por
enxertos, com o nucleo congelado e capacidade nova adicionada em partes.
