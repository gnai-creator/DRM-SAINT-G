# Fase 16 - DRM Full 125M/350M vs SAINT-G Grafted

Status: **implementado, validado em dry-run**.

## Objetivo

Criar uma ponte cientifica antes da escala 70B: treinar um modelo DRM full que
caiba em uma RTX 4090 e comparar contra um modelo DRM menor que cresce por
grafting ate budget equivalente.

Esta fase existe porque a comparacao full em 70B nao e viavel no hardware alvo.
Em vez de exigir um baseline impossivel, a Fase 16 cria um baseline full real em
escala controlada.

## Hipotese

Se SAINT-G e um paradigma util de crescimento por enxerto, entao ele deve
mostrar sinal em uma escala onde ainda conseguimos comparar contra treino full:

```text
DRM full 125M ou 350M
vs
DRM 5M + grafts progressivos ate capacidade/checkpoint equivalente
vs
modelos externos pequenos quando aplicavel
```

O objetivo nao e provar que o grafted vence sempre. O objetivo e medir:

- qualidade por parametro treinavel;
- qualidade por byte de checkpoint;
- estabilidade;
- custo de memoria;
- custo de tempo;
- distancia ate um full model treinado no mesmo dominio.

## Tamanho Alvo

O alvo agora deve seguir os YAMLs reais ja existentes no `drm_transformer`:

```text
drm_transformer/configs/scaling/multilingual/125m.yaml
drm_transformer/configs/scaling/multilingual/350m.yaml
```

O 125M deve ser o primeiro baseline full obrigatorio. O 350M deve ser o segundo
baseline se couber na RTX 4090 com tempo aceitavel.

Para uma RTX 4090, 125M e plausivel para treino experimental. O 350M pode
exigir mais cuidado:

- `bf16` ou `fp16`;
- batch pequeno;
- gradient checkpointing se necessario;
- contexto curto;
- AdamW com cuidado;
- acumulacao de gradiente;
- checkpoints compactos.

Nao vamos criar uma config intermediaria artificial de 250M nesta fase. O Marco
1 deve estimar:

```text
125M
350M
```

O criterio e simples:

```text
o maior DRM full que treina de forma estavel na RTX 4090
sem OOM e com tempo aceitavel
```

## Comparacoes

### Comparacao Principal

```text
DRM full 125M
DRM full 350M, se couber
vs
DRM 5M + SAINT-G grafted ate 125M/350M nominal/funcional
```

Aqui "ate 125M/350M" nao significa necessariamente materializar todos esses
parametros como treinaveis. Pode significar:

- mesmo budget de inferencia;
- mesmo budget de checkpoint;
- mesmo numero efetivo de parametros adicionados;
- mesma familia arquitetural com capacidade nova por enxertos.

O documento de cada experimento deve declarar qual nocao de equivalencia esta
sendo usada.

### Comparacoes Externas

Comparar perplexity contra modelos externos pequenos e util, mas deve ser feito
com cuidado.

Modelos candidatos:

- `facebook/opt-125m`;
- `facebook/opt-350m`;
- `gpt2` pequeno, 124M;
- `gpt2-medium`, 355M;
- outro causal LM local abaixo de 500M.

Observacao:

A comparacao externa deve ser por faixa real:

```text
~125M
~350M
```

Esses modelos nao sao equivalentes ao DRM, mas ajudam a calibrar perplexity,
tempo e memoria.

## Baselines

Baselines obrigatorios:

- DRM full controlado, maior que caiba;
- DRM 5M congelado;
- DRM 5M + Phi graft;
- DRM 5M + Phi multi-stage se disponivel;
- DRM 5M + full-module graft pequeno;
- modelo externo pequeno sem fine-tuning;
- LoRA/QLoRA externo quando couber.

Baselines opcionais:

- DRM full 125M;
- DRM full 350M se couber;
- grafted a partir de 25M/100M, se houver configs intermediarias.

## Marcos

### Marco 1 - Memory Planner 125M/350M

Objetivo:

Determinar o maior DRM full treinavel na RTX 4090.

Entregas:

- estimativa de parametros por config;
- memoria de pesos;
- memoria de gradientes;
- memoria de AdamW;
- memoria de ativacoes;
- estimativa por dtype;
- recomendacao de execucao: 125M obrigatorio, 350M se couber.

Criterio:

Passa se o projeto escolher uma config full viavel antes de treinar.

### Marco 2 - Config DRM Full Controlada

Objetivo:

Criar config DRM full para o tamanho escolhido.

Entregas:

- config em `drm_transformer/configs/scaling/...`;
- estimativa de parametros;
- script de smoke;
- forward/loss em dataset pequeno;
- medicao CUDA de load e forward.

Criterio:

Passa se o modelo full carrega e roda forward sem OOM.

### Marco 3 - Treino Full Curto

Objetivo:

Treinar o DRM full por poucos steps para criar baseline real.

Entregas:

- treino curto;
- loss antes/depois;
- perplexity em validacao;
- memoria CUDA por etapa;
- tokens/s;
- checkpoint full;
- custo de disco.

Criterio:

Passa se o full model melhora loss sem OOM.

### Marco 4 - Grafted 5M ate Budget Alvo

Objetivo:

Partir de `multilingual/5m.yaml` e adicionar capacidade por SAINT-G.

Entregas:

- escolher alvos de enxerto;
- aplicar `phi_zero_full_rank`;
- aplicar `phi_ls_residual`;
- testar Phi multi-stage;
- salvar checkpoint recomponivel;
- medir loss/perplexity;
- medir tamanho do checkpoint.

Criterio:

Passa se o grafted melhora a base 5M e gera checkpoint recomponivel.

Status inicial:

```text
implementado como smoke inicial
script: scripts/benchmark_drm_g_phase16_5m_graft.py
relatorio: docs/reports/phase16_marco4_5m_graft.md
```

Resultado inicial:

- checkpoint base: `drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt`;
- dataset: `drm_transformer/data/multilingual_125m`;
- melhor alvo inicial: `blocks.5.ffn.down_proj`;
- melhor familia media: `phi_ls_full_rank`;
- ganho medio de validacao: `0.000114`;
- positivo em `3/3` seeds;
- parametros treinaveis: `9.216`;
- controle `full_module_linear`: `36.864` parametros e ganho medio negativo.

Interpretacao:

O Marco 4 tem sinal positivo inicial, mas ainda fraco. Ele prova que o caminho
5M + enxerto roda no mesmo dataset e consegue melhorar validacao em um alvo
especifico. Ainda nao prova que o modelo grafted chega perto do full 125M.

### Marco 4B - GraftBlock 5M Repetivel

Objetivo:

Testar explicitamente a ideia:

```text
DRM 5M + enxerto de aproximadamente 5M * 24 ~= DRM 125M
```

Esta variante usa um bloco residual enxertado na saida de blocos do DRM:

```text
h_out = h + scale * down(silu(up(h)))
```

Para o DRM multilingual 5M:

```text
d_model = 96
hidden por enxerto de 5M ~= 26k
24 enxertos ~= 119.3M parametros adicionados
total efetivo ~= 125.0M parametros
```

Status:

```text
implementado como smoke operacional
script: scripts/benchmark_drm_g_phase16_graftblock.py
relatorio: docs/reports/phase16_marco4b_graftblock_5m.md
```

Resultados iniciais:

- 1 enxerto de 5M: ganho de validacao `0.000141`, pico CUDA `381 MB`;
- 2 enxertos de 5M: ganho de validacao `0.000334`, pico CUDA `495 MB`;
- 4 enxertos de 5M: ganho de validacao `0.000811`, pico CUDA `755 MB`;
- 24 enxertos, total efetivo `124,995,595` parametros: ganho de validacao
  `0.000261`, pico CUDA `3.43 GB`.

Interpretacao:

O caminho `5M + 24 enxertos ~= 125M` esta tecnicamente validado em smoke: roda
em CUDA, nao estoura memoria e melhora levemente a validacao. Ainda nao e uma
comparacao de qualidade contra o full 125M, porque o treino foi curto e o
empilhamento ingenuo de 24 enxertos nao superou o caso de 4 enxertos.

Proximo passo:

- treinar 24 enxertos por mais steps;
- fazer crescimento progressivo em vez de ativar todos os enxertos de uma vez;
- salvar checkpoint recomponivel dos enxertos;
- comparar 4/8/16/24 enxertos com o mesmo budget de tokens;
- medir distancia real contra a loss do full 125M.

### Marco 4C - Treinabilidade dos 24 Enxertos

Objetivo:

Transformar o smoke `5M + 24 enxertos` em um experimento comparavel.

Status:

```text
implementado
relatorio: docs/reports/phase16_marco4c_graftblock_training.md
```

Entregas concluidas:

- `--graft-counts 4 8 16 24`;
- treino progressivo por quantidade ativa de enxertos;
- decaimento de learning rate por enxerto;
- warmup de escala por enxerto;
- checkpoint recomponivel;
- avaliacao do checkpoint recarregado;
- distancia para a loss do full 125M smoke.

Resultado principal:

```text
full 125M smoke loss: 9.049912
melhor 4-graft loss: 10.415268
24-graft checkpoint positivo loss: 10.415910
distancia 24-graft -> full 125M: 1.365997
recompose_abs_diff: 0.0
```

Veredito:

O Marco 4C passa na infraestrutura, mas ainda nao passa em qualidade. O caminho
24-graft e recomponivel e cabe em CUDA, mas o treino progressivo ingenuo piorou
validacao quando aumentamos steps. O melhor ponto curto ainda foi 4 enxertos.

Proximo passo:

O Marco 4D deve selecionar e aceitar enxertos por ganho de validacao, congelar os
aceitos e so entao adicionar novos grupos. Nao devemos continuar apenas
empilhando 24 enxertos por indice.

### Marco 4D - Early Stopping e Melhor Checkpoint

Objetivo:

Corrigir o problema observado no treino de 4 horas: o caminho 24-graft e muito
eficiente em VRAM/tempo, mas sem controle de validacao a loss final pode piorar.

Resultado do treino de 4 horas sem early stopping:

```text
base_loss: 10.416174
final_loss: 10.683245
validation_gain: -0.267071
trained_steps: 200.965
cuda_peak: 3.43 GB
recompose_abs_diff: 0.0
```

Status:

```text
implementado e validado em dry-run
relatorio: docs/reports/phase16_marco4d_early_stopping.md
```

Entregas concluidas:

- validacao periodica com `--eval-every-steps`;
- checkpoint do melhor ponto com `--save-best-checkpoint`;
- checkpoint final separado com `--save-graft-checkpoint`;
- parada antecipada com `--early-stopping-patience`;
- delta minimo com `--early-stopping-min-delta`;
- historico em `training_metrics.jsonl`.

Dry-run:

```text
lr: 3e-7
max_train_seconds: 30
trained_steps: 652
final_loss: 10.415352
final_gain: 0.000822
best_eval_loss: 10.415417
best_eval_gain: 0.000757
best_recompose_abs_diff: 0.0
```

Veredito:

O Marco 4D passa na infraestrutura de controle de treino. As proximas
comparacoes de qualidade devem usar `best_eval_loss` e o artefato
`best_graft_checkpoint`, nao apenas `final_loss`.

Resultado adicional 4-graft:

```text
graft_count: 4
trainable_parameters: 19.882.756
cuda_peak: 746 MB
best_eval_step: 5.000
best_eval_loss: 10.414828
best_eval_gain: 0.001346
best_checkpoint_size: 79.536.323 bytes
best_recompose_abs_diff: 0.0
```

Esse resultado confirma que grupos menores treinam melhor que 24 enxertos
simultaneos. O ponto final do treino ainda piorou, mas o melhor checkpoint foi
salvo e recomposto exatamente.

### Marco 4E - Staged Graft Growth

Status: **implementado, validado em dry-run**.

Objetivo:

Treinar crescimento por estagios aceitos, em vez de ativar 24 enxertos de uma
vez.

Procedimento:

```text
1. treinar G1 com 4 enxertos
2. aceitar G1 somente se best_eval_gain > 0
3. congelar G1 aceito
4. adicionar G2 com mais 4 enxertos
5. treinar G2 com G1 ativo e congelado
6. aceitar G2 somente se nao destruir G1
7. repetir ate 24 enxertos
```

Motivacao:

Os resultados atuais sugerem:

```text
24 enxertos simultaneos -> instavel ou sem ganho
4 enxertos com best checkpoint -> ganho positivo
```

Logo, o caminho correto e crescimento incremental aprovado por validacao.

Entregas:

- script ou modo para treinar grupos de 4 enxertos por estagio;
- checkpoint composto com todos os grupos aceitos;
- politica `approve/reject/defer`;
- congelamento de grupos aceitos;
- validacao apos cada estagio;
- relatorio G1, G1+G2, G1+G2+G3 ... ate 24;
- comparacao contra melhor 4-graft isolado.

Implementado:

```text
--training-mode staged
--stage-size
--max-stages
--freeze-accepted-stages
--stage-accept-min-gain
```

Artefatos:

```text
stage_metrics.json
stage_training_metrics.jsonl
composed_graft_checkpoint.pt
summary.json
results.md
```

Dry-run:

```text
runs/phase16_marco4e_staged_dryrun
recompose_abs_diff: 0.0
```

Resultado 24-graft staged:

```text
runs/phase16_marco4e_staged_24graft
base_loss: 10.416174
composed_loss: 10.414818
accumulated_gain: 0.001357
accepted_stages: 1
accepted_grafts: 4
cuda_peak: 1.30 GB
recompose_abs_diff: 0.0
```

Decisoes:

```text
stage 1, grafts 0-3: approved, gain 0.001357
stage 2, grafts 4-7: rejected, gain 0.000000
```

Interpretacao:

O Marco 4E se comportou como esperado: aceitou o grupo que melhorou validacao,
rejeitou o grupo que nao acrescentou ganho e preservou o checkpoint composto. O
ganho acumulado ficou ligeiramente acima do melhor 4-graft isolado.

Criterio:

```text
G1 melhora validacao
G1+G2 nao destroi G1
checkpoint composto recompoe
ganho acumulado > melhor 4-graft isolado
```

### Marco 4F - Validation-Routed Staged Grafts

Status: **pendente**.

Objetivo:

Escolher os proximos grupos de enxertos por ganho de validacao, nao por ordem
fixa de indice.

Motivacao:

O Marco 4E aprovou `grafts 0-3` e rejeitou `grafts 4-7`. Isso indica que o
problema agora e selecao de candidatos. O proximo grupo nao deve ser simplesmente
o proximo indice; deve ser o grupo que demonstra maior ganho em validacao.

Procedimento:

```text
1. gerar candidatos em blocks.0..5
2. treinar cada candidato por budget curto
3. medir loss do modelo composto com o candidato
4. rankear candidatos por composed validation gain
5. aceitar o top grupo se composed gain > threshold
6. congelar grupo aceito
7. repetir ate budget alvo ou sem candidatos positivos
```

Criterio:

```text
selecionar grupo melhor que ordem fixa
ganho acumulado > Marco 4E
checkpoint composto recompoe
VRAM permanece controlada
```

Implementado:

```text
--training-mode validation_routed_staged
--candidate-targets
```

Artefatos:

```text
candidate_metrics.json
candidate_training_metrics.jsonl
stage_metrics.json
composed_graft_checkpoint.pt
summary.json
results.md
```

Dry-run:

```text
runs/phase16_marco4f_routed_dryrun
candidatos testados: blocks.1, blocks.2
accepted_groups: 0
accumulated_gain: 0.0
```

O dry-run validou o caminho de runtime: candidatos foram testados, ranqueados,
rejeitados quando nao melhoraram validacao e os artefatos esperados foram
gerados.

Marco 4F-fix:

```text
status: implementado
motivo: a primeira versao aceitava por ganho local do candidato
correcao: aceitar somente se candidate_composed_loss melhora o checkpoint composto
estado: candidate_composed_loss usa o best_state_payload do candidato
candidate_metrics.json: salva previous_composed_loss, candidate_composed_loss,
candidate_composed_gain e candidate_target_by_graft
checkpoint: salva target_by_graft para recompor cada grupo no bloco escolhido
dry-run: runs/phase16_marco4f_fix_dryrun
recompose_abs_diff: 0.0
```

O run 24-graft anterior continua valido como diagnostico: o roteador executou e
recompos exatamente, mas aceitou grupos que melhoravam probes locais e pioravam
o modelo composto. A regra corrigida transforma o aceite em uma decisao global.

Resultado 24-graft corrigido:

```text
runs/phase16_marco4f_best_payload_24graft
base_loss: 10.416174
composed_loss: 10.414808
accumulated_gain: 0.001366
accepted_groups: 1
accepted_grafts: 4
selected_target: blocks.2
stage 2: rejected, composed_gain 0.0
recompose_abs_diff: 0.0
```

Veredito:

```text
Marco 4F passou.
```

O roteador agora escolhe `blocks.2` automaticamente, usa o melhor estado de
validacao do candidato, aceita apenas ganho composto real e preserva checkpoint
recomponivel. O proximo gargalo e diversidade de candidatos apos G1.

### Marco 4G - Candidate Grid Routed Growth

Status: **implementado, validado em dry-run**.

Objetivo:

Tentar aprovar mais de um grupo de enxertos usando uma busca de candidatos mais
diversa.

Procedimento:

```text
1. manter aceite por composed validation gain
2. testar target x learning_rate x init_scale x activation
3. aceitar o melhor candidato positivo
4. congelar o grupo aceito
5. repetir ate 24 grafts ou ate nao haver candidato positivo
```

Criterio:

```text
ganho acumulado > Marco 4F
ou
mais de um grupo aceito sem regressao composta
checkpoint composto recompoe
VRAM permanece controlada
```

Implementado:

```text
--candidate-learning-rates
--candidate-init-scales
--candidate-activations
```

Dry-run:

```text
runs/phase16_marco4g_grid_dryrun
candidatos: 4
accepted_groups: 0
accumulated_gain: 0.0
recompose_abs_diff: 0.0
```

Documento:

```text
docs/reports/phase16_marco4g_candidate_grid_routed_grafts.md
```

Resultado light probe:

```text
runs/phase16_marco4g_light_probe_24graft
base_loss: 10.416174
composed_loss: 10.414729
accumulated_gain: 0.001446
accepted_groups: 1
accepted_grafts: 4
selected_target: blocks.2
learning_rate: 1e-7
recompose_abs_diff: 0.0
```

Veredito:

```text
Marco 4G passou como melhoria incremental.
```

### Marco 4H - Fine-Grained Second Stage

Status: **implementado, validado em dry-run**.

Objetivo:

Verificar se G2 falha por falta de ganho restante ou por granularidade grossa
demais com `stage_size=4`.

Procedimento:

```text
1. manter G1 com a melhor configuracao do Marco 4G
2. para G2 em diante, reduzir stage_size para 1 ou 2
3. testar grid local de targets/lr/init_scale
4. aceitar somente por composed validation gain
5. parar apos candidatos sem ganho positivo
```

Criterio:

```text
composed_loss < 10.414729
accepted_grafts > 4
recompose_abs_diff = 0.0
```

Implementado:

```text
--post-first-stage-size
```

Dry-run:

```text
runs/phase16_marco4h_adaptive_stage_dryrun
marco: 4h_fine_grained_second_stage
stage 1 size: 2
stage 2 size: 1
recompose_abs_diff: 0.0
```

Documento:

```text
docs/reports/phase16_marco4h_fine_grained_second_stage.md
```

Resultado 24-graft:

```text
runs/phase16_marco4h_fine_g2_24graft
base_loss: 10.416174
composed_loss: 10.414671
accumulated_gain: 0.001504
accepted_groups: 2
accepted_grafts: 5
stage 1: blocks.2, grafts 0-3, gain 0.001450
stage 2: blocks.3, graft 4, gain 0.000054
stage 3: rejected
recompose_abs_diff: 0.0
```

Veredito:

```text
Marco 4H passou.
```

O resultado confirma que G2 nao era impossivel. O `stage_size=4` era grosso
demais depois de G1. Com granularidade 1, o roteador aceitou um enxerto
incremental em `blocks.3` sem regressao composta.

### Marco 4I - Residual/Orthogonal Routing

Status: **implementado, validado em dry-run**.

Objetivo:

Testar se o estagio 3 falha porque candidatos posteriores repetem targets ja
explorados.

Implementado:

```text
--candidate-score-mode composed_gain
--candidate-score-mode composed_gain_orthogonal
--orthogonal-penalty
```

Regra:

```text
candidate_score = candidate_composed_gain - redundancy_penalty
redundancy_penalty = orthogonal_penalty * accepted_grafts_on_same_target
```

O aceite continua estrito:

```text
candidate_composed_gain > stage_accept_min_gain
```

Dry-run:

```text
runs/phase16_marco4i_orthogonal_strict_dryrun
candidate_score_mode: composed_gain_orthogonal
stage_gain: 0.0
decision: rejected
recompose_abs_diff: 0.0
```

Criterio:

```text
composed_loss < 10.414671
accepted_grafts > 5
recompose_abs_diff = 0.0
```

Documento:

```text
docs/reports/phase16_marco4i_residual_orthogonal_routing.md
```

Resultado 24-graft:

```text
runs/phase16_marco4i_light_orthogonal
base_loss: 10.416174
composed_loss: 10.414714
accumulated_gain: 0.001460
accepted_groups: 2
accepted_grafts: 5
stage 1: blocks.2, grafts 0-3
stage 2: blocks.3, graft 4
stage 3: rejected with redundancy control
recompose_abs_diff: 0.0
```

Veredito:

```text
Marco 4I passou como controle de redundancia, mas nao bateu o Marco 4H.
```

### Marco 4J - Two-Pass Candidate Pruning

Status: **concluido**.

Objetivo:

Reduzir o custo do grid completo de candidatos. Em vez de treinar profundamente
todos os candidatos, o Marco 4J faz uma passagem barata de probe, ranqueia os
candidatos, mantem apenas `top-k` e treina profundamente apenas esses.

Implementado:

```text
--candidate-probe-steps
--candidate-probe-max-train-seconds
--candidate-top-k
```

Documento:

```text
docs/reports/phase16_marco4j_two_pass_candidate_pruning.md
```

Resultado CUDA:

```text
runs/phase16_marco4j_two_pass_24graft
base_loss: 10.416174
composed_loss: 10.414808
accumulated_gain: 0.001366
accepted_groups: 1
accepted_grafts: 4
stage 1: blocks.2, grafts 0-3, approved
stage 2: blocks.4, graft 4, rejected
recompose_abs_diff: 0.0
```

Comparacao:

```text
4H composed_loss: 10.414671, accepted_grafts: 5
4I composed_loss: 10.414714, accepted_grafts: 5
4J composed_loss: 10.414808, accepted_grafts: 4
```

Veredito:

```text
Marco 4J passou tecnicamente, mas nao como melhor qualidade.
```

O two-pass routing funcionou e recompos exatamente, mas `candidate_top_k=4` com
probe curto perdeu o quinto graft encontrado pelo 4H em `blocks.3`.

### Marco 4K - Two-Pass Top-K 8 Probe 2K

Status: **concluido / novo melhor checkpoint grafted da Fase 16**.

Objetivo:

Recuperar o quinto graft do Marco 4H usando a infraestrutura two-pass do Marco
4J, mas com pruning menos agressivo.

Ajustes contra 4J:

```text
candidate_targets: blocks.0..5 -> blocks.2 blocks.3 blocks.4
candidate_probe_steps: 1000 -> 2000
candidate_probe_max_train_seconds: 180 -> 300
candidate_top_k: 4 -> 8
```

Documento:

```text
docs/reports/phase16_marco4k_two_pass_topk8_probe2k.md
```

Resultado CUDA:

```text
runs/phase16_marco4k_two_pass_topk8_probe2k_24graft
base_loss: 10.416174411773682
composed_loss: 10.414523839950562
recomposed_loss: 10.414523839950562
recompose_abs_diff: 0.0
accumulated_gain: 0.0016505718231201172
accepted_groups: 2
accepted_grafts: 5
checkpoint: composed_graft_checkpoint.pt
checkpoint_bytes: 477209927
```

Stage decisions:

```text
stage 1: approved, grafts 0-3 -> blocks.4, gain 0.001550
stage 2: approved, graft 4 -> blocks.2, gain 0.000101
stage 3: rejected, blocks.3, gain 0.0
```

Comparacao:

```text
4H composed_loss: 10.414671, accepted_grafts: 5
4I composed_loss: 10.414714, accepted_grafts: 5
4J composed_loss: 10.414808, accepted_grafts: 4
4K composed_loss: 10.414524, accepted_grafts: 5
```

Veredito:

```text
Marco 4K passou e virou o novo melhor resultado grafted da Fase 16.
```

O 4K recuperou um quinto graft util e bateu o 4H, 4I e 4J. A rota final nao foi
a hipotese original `blocks.3`; o melhor checkpoint usou `blocks.4` para os
quatro primeiros grafts e `blocks.2` para o quinto. Ainda assim, a recomposicao
foi exata e o resultado e o melhor checkpoint grafted ate aqui.

Observacao: o `summary.json` gerado ainda rotula `marco` como
`4j_two_pass_candidate_pruning`, porque o helper `_marco_name()` atualmente
classifica qualquer run com `candidate_top_k > 0` como 4J. O diretorio, comando e
relatorio identificam corretamente este experimento como Marco 4K.

### Marco 4L - Robustez do 4K em Multiplas Seeds

Status: **planejado / proximo CUDA run recomendado**.

Objetivo:

Verificar se o 4K e um ganho robusto ou um resultado especifico da seed 42 antes
de promover o checkpoint/receita para a comparacao full-vs-grafted.

Documento:

```text
docs/reports/phase16_marco4l_4k_multiseed_robustness.md
```

Plano:

```text
replicar a receita 4K em seeds adicionais: 7 e 123
rodar um comando por seed, com output-dir separado
manter candidate_targets, candidate_probe_steps, candidate_top_k e criterio de aceite
comparar composed_loss, accepted_grafts e target_by_graft por seed
validar recompose_abs_diff = 0.0 em todos os checkpoints
corrigir ou documentar o label `_marco_name()` para runs top-k 4K+
```

Comandos principais:

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/benchmark_drm_g_phase16_graftblock.py \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4l_two_pass_topk8_probe2k_24graft_seed7 \
  --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
  --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
  --device cuda \
  --seeds 7 \
  --graft-count 24 \
  --hidden-size 25889 \
  --stage-size 4 \
  --post-first-stage-size 1 \
  --max-stages 8 \
  --stage-accept-min-gain 0.0 \
  --steps 100000000 \
  --max-train-seconds 1800 \
  --eval-every-steps 5000 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.00001 \
  --batch-size 2 \
  --seq-len 128 \
  --validation-batches 4 \
  --train-batches 4096 \
  --learning-rate 0.0000003 \
  --lr-decay 0.02 \
  --training-mode validation_routed_staged \
  --candidate-targets blocks.2 blocks.3 blocks.4 \
  --candidate-learning-rates 0.00000003 0.0000001 0.0000003 \
  --candidate-init-scales 0.001 0.005 0.01 \
  --candidate-activations silu \
  --candidate-score-mode composed_gain_orthogonal \
  --orthogonal-penalty 0.00001 \
  --candidate-probe-steps 2000 \
  --candidate-probe-max-train-seconds 300 \
  --candidate-top-k 8

python \
  scripts/benchmark_drm_g_phase16_graftblock.py \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4l_two_pass_topk8_probe2k_24graft_seed123 \
  --checkpoint /mnt/e/dev/ai/drm_transformer/checkpoints/multilingual_5m/smoke_819k/final.pt \
  --data-dir /mnt/e/dev/ai/drm_transformer/data/multilingual_125m \
  --device cuda \
  --seeds 123 \
  --graft-count 24 \
  --hidden-size 25889 \
  --stage-size 4 \
  --post-first-stage-size 1 \
  --max-stages 8 \
  --stage-accept-min-gain 0.0 \
  --steps 100000000 \
  --max-train-seconds 1800 \
  --eval-every-steps 5000 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.00001 \
  --batch-size 2 \
  --seq-len 128 \
  --validation-batches 4 \
  --train-batches 4096 \
  --learning-rate 0.0000003 \
  --lr-decay 0.02 \
  --training-mode validation_routed_staged \
  --candidate-targets blocks.2 blocks.3 blocks.4 \
  --candidate-learning-rates 0.00000003 0.0000001 0.0000003 \
  --candidate-init-scales 0.001 0.005 0.01 \
  --candidate-activations silu \
  --candidate-score-mode composed_gain_orthogonal \
  --orthogonal-penalty 0.00001 \
  --candidate-probe-steps 2000 \
  --candidate-probe-max-train-seconds 300 \
  --candidate-top-k 8
```

Criterio:

```text
passa se a media multi-seed continuar melhor que o 4H
ou se todas as seeds preservarem 5 accepted_grafts com recomposicao exata.
```

Se o 4L passar, o Marco 5 deve usar a receita 4K/4L como melhor representante
grafted contra o baseline full controlado. Como o resultado observado ate agora
indica que o quinto graft nao e robusto em todas as seeds, os Marcos 4M/4N foram
inseridos antes do Marco 5 para diagnosticar e possivelmente melhorar o routing.

### Marco 4M - NTK-Mirror-Inspired Activation Gate Probe

Status: **implementado / runs diagnosticos CUDA em andamento**.

Objetivo:

Implementar um probe NTK-style no SAINT-G para ranquear `blocks.2`, `blocks.3` e
`blocks.4` antes do treino deep dos graft candidates.

Documento:

```text
docs/reports/phase16_marco4m_ntkmirror_activation_gate_probe.md
```

Sinal calculado:

```text
score(block) = sum(abs(grad_h * h))
```

Entregas implementadas:

```text
--ntk-activation-probe-batches N
--ntk-activation-probe-split train|val
ntk_activation_probe_metrics.json
stage_metrics[*].ntk_activation_probe
```

Escopo minimo:

```text
1. adicionar captura de ativacoes + gradientes nos candidate targets
2. calcular score por target com abs(grad_h * h)
3. comparar ranking do score com candidates escolhidos em seeds 42, 7 e 123
4. verificar se o score prediz stage aprovado/rejeitado, melhor target e quinto graft
```

Criterio:

```text
O score NTK-style explica ou prediz por que seed 42 encontra o quinto graft
e seeds 7/123 nao.
```

Marco 4M e diagnostico somente; ele nao substitui o `composed_gain_orthogonal`
ainda.

Status operacional atual:

```text
seed 42: completed; reproduced 4K best result
  composed_loss: 10.414523839950562
  accepted_grafts: 5
  route: grafts 0-3 -> blocks.4, graft 4 -> blocks.2
  selected top-1 rate in 4N-A: 0.333

seed 7: completed
  composed_loss: 10.386313915252686
  accepted_grafts: 4
  route: grafts 0-3 -> blocks.4
  stage 2 selected blocks.2 and rejected it with gain 0.0
  selected top-1 rate in 4N-A: 0.500

seed 123: completed
  composed_loss: 10.415361166000366
  accepted_grafts: 4
  route: grafts 0-3 -> blocks.3
  stage 1 selected blocks.3 even though raw NTK top-1 was blocks.4
  stage 2 selected blocks.4 and rejected it with gain 0.0
  selected top-1 rate in 4N-A: 0.500
```

Final 4M/4N-A interpretation: raw NTK score is a useful global sensitivity
diagnostic, but not a safe routing signal. Seed 42 approved stage-2 `blocks.2`
even though raw NTK ranked it last; seed 7 rejected a similar `blocks.2` stage;
seed 123 got seed-42-scale gain from `blocks.3` even though raw NTK ranked
`blocks.4` first. Marco 4N remains split into 4N-A and 4N-B: 4N-A completed the
offline residual/saturation analysis, and 4N-B should be a conservative routing
experiment only if it preserves known useful targets.

### Marco 4N - NTK Residual/Saturation Routing Plan

Status: **4N-A concluido para seeds 42, 7 e 123 / 4N-B implementado como experimento conservador**.

Objetivo:

Dividir o Marco 4N em duas etapas:

```text
4N-A:
  analise offline residual/saturation-aware dos artifacts 4M
  sem alterar roteamento ou treino

4N-B:
  experimento de routing NTK-aware conservador implementado
  composto por saturation-adjusted NTK, residual-delta NTK e anti-saturation
  sem descartar representantes raw NTK rank 2/rank 3
```

Documento:

```text
docs/reports/phase16_marco4n_ntk_guided_candidate_routing.md
```

Implementacao 4N-A:

```text
saint/adapters/drm_grafting_ntk_analysis.py
scripts/analyze_phase16_ntk_residual_saturation.py
tests/test_ntk_residual_saturation_analysis.py
```

Comando 4N-A completo para seeds 42, 7 e 123:

```bash
cd /home/rato/dev/ai/SAINT-G

python \
  scripts/analyze_phase16_ntk_residual_saturation.py \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed42 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed7 \
  --run-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4m_ntk_probe_topk8_probe2k_24graft_seed123 \
  --output-dir /home/rato/dev/ai/SAINT-G/runs/phase16_marco4n_a_ntk_residual_saturation_seed42_seed7_seed123
```

Resultado 4N-A:

```text
wrote 21 joined rows
seed=42  recommendations=reject_raw_ntk_prefilter,test_saturation_adjusted_ntk,test_residual_delta_ntk,include_target_saturation_features
seed=7   recommendations=reject_raw_ntk_prefilter,test_saturation_adjusted_ntk,test_residual_delta_ntk,include_target_saturation_features
seed=123 recommendations=reject_raw_ntk_prefilter,test_saturation_adjusted_ntk,test_residual_delta_ntk,include_target_saturation_features
```

Saidas 4N-A:

```text
ntk_candidate_joined_metrics.json
ntk_stage_feature_table.csv
ntk_run_summaries.json
ntk_routing_analysis.md
```

Implementacao 4N-B:

```text
saint/adapters/drm_grafting_graftblock_routed_utils.py
saint/adapters/drm_grafting_graftblock_routed.py
scripts/benchmark_drm_g_phase16_graftblock.py
tests/test_ntk_hybrid_routing.py
```

Modo 4N-B:

```text
--candidate-score-mode composed_gain_ntk_hybrid_conservative
```

Regra implementada:

```text
saturation_adjusted_ntk = ntk_score / (1 + accepted_grafts_on_target)
residual_delta_ntk = abs(ntk_score_stage_t - ntk_score_stage_t_minus_1)
anti_saturation_penalty = weight * accepted_grafts_on_target

candidate_score = candidate_composed_gain
                - orthogonal_penalty
                - anti_saturation_penalty
                + saturation_weight * saturation_adjusted_ntk
                + residual_delta_weight * residual_delta_ntk
```

Seguranca implementada:

```text
candidate_top_k ainda controla os melhores candidatos por score,
mas --ntk-hybrid-keep-ranks 3 adiciona pelo menos um representante dos raw NTK
ranks 1, 2 e 3 quando existirem. Isso evita descartar seed 42 stage-2 blocks.2
(rank 3) e seed 123 stage-1 blocks.3 (rank 2).
```

### Marco 4O - Tensor-Network Follow-ups from ITensors.jl

Status: **planejado / design documentado**.

Objetivo:

Documentar e transformar ideias do ITensors.jl / ITensorMPS.jl em marcos
PyTorch-native para SAINT-G e drm_transformer, sem adicionar Julia como
dependencia imediata.

Documento:

```text
docs/reports/phase16_marco4o_tensor_network_followups.md
```

Motivacao:

```text
ITensors.jl sugere disciplina de tensor networks:
- indices nomeados/taggeados;
- SVD/truncated-SVD com erro controlado;
- MPS/MPO/Tensor Train com bond dimension explicito;
- custo de contracao;
- block-sparse/QN como analogia para setores de canais.
```

Submarcos recomendados:

```text
4O-lite - Graft SVD Anatomy:
  carregar composed_graft_checkpoint.pt,
  extrair matrizes dos grafts aceitos,
  medir espectro singular, rank efetivo e erro de truncamento,
  comparar seeds 42/7/123 e targets blocks.2/3/4.

4O - Tensor-Train / MPS Adapter Baseline:
  implementar TTLinear ou TTGraftBlock em PyTorch,
  varrer bond_dim chi = 2/4/8/16,
  comparar contra graft blocks em loss, parametros, bytes, runtime e robustez.

DRM Marco A - Manifold Attention Tensor Anatomy:
  no repo drm_transformer, capturar tensores de attention geometrica,
  medir rank/entanglement/compressibilidade por layer/head/token/manifold_dim.
```

Criterio:

```text
passa se 4O-lite mostrar se os grafts aceitos usam capacidade real ou sao
compressiveis para low-rank / Tensor Train pequeno. Se compressiveis, implementar
4O completo; se nao, manter graft blocks e usar SVD apenas como diagnostico.
```

Relacao com 4M/4N:

```text
4M/4N respondem onde adaptar.
4O-lite responde quanta capacidade os grafts uteis realmente precisam.
4O testa se essa capacidade pode ser representada por Tensor Train / MPS.
```

### Marco 5 - Comparacao Full vs Grafted

Objetivo:

Comparar full controlado contra grafted em metricas honestas.

Metricas:

- validation loss;
- perplexity;
- ganho por parametro treinavel;
- ganho por byte de checkpoint;
- memoria CUDA maxima;
- tempo por step;
- tokens/s;
- estabilidade por seed;
- regressao em exemplos antigos.

Criterio:

Passa se o relatorio deixa claro onde grafting ganha e onde perde.

### Marco 6 - Comparacao Externa

Objetivo:

Comparar perplexity e custo contra modelos externos pequenos.

Entregas:

- avaliar `gpt2` 124M;
- avaliar `gpt2-medium` 355M se couber;
- avaliar `opt-125m`;
- avaliar `opt-350m` se couber;
- mesma tokenizacao/corpus quando possivel;
- relatorio de incompatibilidades de tokenizer.

Criterio:

Passa se a comparacao externa for interpretavel, mesmo que nao seja perfeitamente
equivalente ao DRM.

### Marco 7 - Decisao de Entrada para 70B

Objetivo:

Decidir se vale mover a escala 70B para a fase seguinte.

Perguntas:

- Grafted chega perto do full controlado em perplexity?
- Grafted vence em checkpoint/memoria?
- O custo de tempo e aceitavel?
- O metodo e estavel em mais de uma seed?
- A vantagem permanece em dataset maior?

Criterio:

Passa se houver recomendacao clara:

```text
avancar para Fase 17 - Prova Experimental Publicavel e Controle
ou
voltar para melhorar Phi/runtime
```

## Criterio de Sucesso da Fase

Sucesso minimo:

```text
treinar DRM full controlado na RTX 4090
treinar DRM 5M + grafts
comparar loss/perplexity/memoria/checkpoint
gerar relatorio final
```

Sucesso forte:

```text
DRM 5M + grafts fica competitivo contra DRM full controlado
em perplexity por byte, memoria ou parametro treinavel
```

Falha:

- nenhum DRM full controlado cabe;
- grafted nao melhora a base 5M;
- checkpoint grafted nao recompõe;
- perplexity fica muito pior sem vantagem de memoria/checkpoint;
- comparacao externa fica inutil por incompatibilidade de dados/tokenizer.

## Veredito Sobre a Ideia

A ideia e boa e cientificamente melhor que saltar direto para 70B.

Ela cria uma pergunta testavel:

```text
Se eu tenho hardware para treinar full 125M e talvez 350M,
SAINT-G consegue crescer de 5M para capacidade parecida
com melhor eficiencia?
```

Isso cria um baseline real. Depois, quando formos para 70B, teremos uma curva de
extrapolacao, nao apenas uma demonstracao isolada.

## Proximo Passo Imediato

Executar o Marco 4L:

- replicar a receita 4K em seeds adicionais;
- confirmar se o ganho contra 4H e robusto;
- validar `recompose_abs_diff = 0.0` em todos os checkpoints;
- corrigir/documentar o label `_marco_name()` para diferenciar 4J/4K/4L;
- se 4L passar, promover a receita 4K/4L para o Marco 5 como melhor representante grafted.
