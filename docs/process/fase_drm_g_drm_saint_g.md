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

Resultado inicial:

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

- aumentar a escala dos textos reais alem de `batch_size=4` e `validation_batches=3`;
- aumentar o corpus alem do fixture tokenizado atual;
- testar mais modelos DRM crescidos;
- salvar artefato `.pt` consolidado, nao apenas validar merge em memoria.

Benchmark multiseed:

```text
script: scripts/benchmark_drm_g_marco4.py
seeds: 31, 32, 33, 34
sequencias: activation_stack, linear_stack
validation_batches: 3
batch_size: 4
```

Resumo:

| metrica | valor |
|---|---:|
| runs | 8 |
| runs positivos | 4 |
| runs com 2+ enxertos aprovados | 2 |
| melhor sequence_gain | 0.000495 |
| melhor gain/param | 3.8669e-06 |
| checkpoint melhor run | 78786 bytes |
| cuda_peak_bytes CPU run | 0 |
| routing_s CPU run | 0.2380 |
| train_s CPU run | 0.5528 |
| eval_s CPU run | 1.2775 |

Melhor run:

| seed | sequencia | aprovados | rejeitados | adiados | old_regression |
|---:|---|---:|---:|---:|---:|
| 33 | `linear_stack` | 2 | 0 | 1 | -0.000495 |

Etapas do melhor run:

| etapa | alvo | init | validation_gain | dense_gain | decisao |
|---:|---|---|---:|---:|---|
| 1 | `blocks.1.attn.out_proj` | `gradient` | 0.000293 | -0.040218 | approve |
| 2 | `blocks.2.attn.out_proj` | `gradient` | 0.000202 | -0.041443 | approve |
| 3 | `blocks.3.attn.out_proj` | `gradient` | -0.000020 | -0.039980 | defer |

O checkpoint recomposto da melhor sequencia reproduziu a media de validacao:

```text
base_loss: 10.805441
graft_loss: 10.804946
validation_gain: 0.000495
```

Merge permanente no `state_dict`:

```text
config: configs/drm_g_marco4_eval_linear_progressive_real_tokens.json
eval_state_merge: true
merge_loss_abs_diff: 0.0
state_dict_merge_supported: true
merged_graft_loss: 10.804946
```

CUDA:

```text
config: configs/drm_g_marco4_linear_progressive_cuda.json
approved_grafts: 3
sequence_gain: 0.000379
cuda_routing_peak_bytes: 71483904
cuda_train_peak_bytes: 51806720
cuda_eval_peak_bytes: 45419520
cuda_peak_bytes: 45419520
```

Leitura atualizada:

O Marco 4 agora cobre as pendencias principais: multiseed, mais exemplos reais,
fila `approve/reject/defer`, alvos lineares consolidaveis, tres candidatos por
fila, metrica de conflito, tamanho de checkpoint, tempo separado e regressao em
exemplos antigos. A sequencia linear tambem foi consolidada em memoria no
`state_dict` e reproduziu exatamente a loss via hook. A medicao CUDA passou no
DRM 3.5M com pico abaixo de 72 MB neste smoke.

### Marco 5 - Validacao Robusta e Artefato Consolidado

Status: **iniciado**.

Objetivo:

Transformar o ciclo progressivo em um fluxo reproduzivel:

```text
treinar enxerto -> aprovar -> consolidar -> salvar .pt -> recarregar -> avaliar -> decidir
```

#### Marco 5A - Artefato Consolidado em Disco

Objetivo:

Persistir a consolidacao linear em um checkpoint PyTorch real do DRM.

Entregas:

- carregar DRM base;
- carregar `drm_graft_sequence_payload`;
- aplicar merge permanente no `state_dict`;
- salvar `consolidated_model.pt`;
- recarregar o `.pt`;
- avaliar o modelo salvo;
- comparar loss do hook, loss do merge em memoria e loss do `.pt` salvo.

Criterio:

Passa se:

- `merge_loss_abs_diff <= 1e-6`;
- `saved_loss_abs_diff <= 1e-6`;
- o arquivo `.pt` existe;
- o manifest ou relatorio registra caminho, bytes e checksum.

Resultado inicial:

```text
config: configs/drm_g_marco5a_consolidate_linear.json
artifact: runs/drm_g_marco5a_consolidate_linear/consolidated_model.pt
artifact_bytes: 13902349
artifact_sha256: 68544e26197a4a83b1c3789cdd2dc92d599b745213f25e48ad8da41d73e8642e
base_loss: 10.805441
hook_loss: 10.804946
saved_loss: 10.804946
saved_loss_abs_diff: 0.0
state_dict_merge_supported: true
```

Veredito:

5A passou no criterio inicial: o artefato `.pt` consolidado foi salvo,
recarregado e avaliou com a mesma loss do caminho por hook.

#### Marco 5B - Retencao e Dados Maiores

Objetivo:

Reduzir o risco de overfit ao fixture curto.

Entregas:

- aumentar `validation_batches`;
- usar offsets separados para treino, validacao e retencao;
- medir `old_regression` em mais batches;
- salvar tabela por seed;
- reprovar enxerto que melhora validacao nova mas degrada retencao.

Criterio:

Passa se `sequence_gain > 0` e `old_regression` fica dentro do limite configurado.

#### Marco 5C - Baseline Full Mais Forte

Objetivo:

Comparar contra um controle treinavel mais honesto.

Entregas:

- full-budget linear com mais steps;
- full fine-tuning pequeno do mesmo modulo/camada;
- comparacao por parametro treinavel;
- comparacao por bytes de checkpoint;
- comparacao por tempo.

Criterio:

Passa se DRM-SAINT-G vencer pelo menos um eixo relevante:

- maior ganho por parametro;
- menor checkpoint;
- menor memoria;
- melhor retencao;
- ganho positivo quando full-budget falha.

#### Marco 5D - Segundo Tamanho DRM

Objetivo:

Testar se o resultado nao e exclusivo do DRM 3.5M.

Entregas:

- detectar configs em `drm_transformer/configs/baselines`;
- selecionar pelo menos um segundo tamanho viavel;
- repetir o ciclo progressivo linear;
- registrar diferencas de perda, memoria e tempo.

Criterio:

Passa se o runtime executa em outro tamanho e produz metricas comparaveis, mesmo
que a qualidade ainda nao supere o melhor run 3.5M.

#### Marco 5E - Criterio Automatico Final

Objetivo:

Transformar o veredito da fase em regra programavel.

Entregas:

- funcao `evaluate_drm_g_phase_success`;
- JSON `phase_decision`;
- limites configuraveis:
  - ganho minimo;
  - regressao maxima;
  - diferenca maxima hook/merge;
  - budget CUDA maximo;
  - minimo de seeds positivos;
- Markdown com aprovado/reprovado e razoes.

Criterio:

Passa se a decisao automatica bater com a leitura manual do benchmark.

#### Marco 5F - Relatorio Final DRM-G

Objetivo:

Fechar a fase DRM-G com uma recomendacao tecnica.

Entregas:

- resumo dos Marcos 1 a 5;
- melhor configuracao atual;
- limites conhecidos;
- recomendacao para proxima fase;
- lista de riscos antes de escalar.

Criterio:

Passa se o projeto tiver resposta clara para:

- DRM-G deve avancar?
- Qual baseline carregar para a proxima fase?
- O foco deve ser qualidade, escala ou infraestrutura?

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
