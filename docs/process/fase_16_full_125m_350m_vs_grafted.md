# Fase 16 - DRM Full 125M/350M vs DRM-SAINT-G Grafted

Status: **implementado, validado em dry-run**.

## Objetivo

Criar uma ponte cientifica antes da escala 70B: treinar um modelo DRM full que
caiba em uma RTX 4090 e comparar contra um modelo DRM menor que cresce por
grafting ate budget equivalente.

Esta fase existe porque a comparacao full em 70B nao e viavel no hardware alvo.
Em vez de exigir um baseline impossivel, a Fase 16 cria um baseline full real em
escala controlada.

## Hipotese

Se DRM-SAINT-G e um paradigma util de crescimento por enxerto, entao ele deve
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
DRM 5M + DRM-SAINT-G grafted ate 125M/350M nominal/funcional
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

Partir de `multilingual/5m.yaml` e adicionar capacidade por DRM-SAINT-G.

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

Status: **pendente**.

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
3. medir best_eval_gain
4. rankear candidatos
5. aceitar o top grupo se gain > threshold
6. congelar grupo aceito
7. repetir ate budget alvo ou sem candidatos positivos
```

Criterio:

```text
selecionar grupo melhor que ordem fixa
ganho acumulado > Marco 4E
checkpoint composto recompÃµe
VRAM permanece controlada
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
avancar para Fase 17 - Escala 70B
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
DRM-SAINT-G consegue crescer de 5M para capacidade parecida
com melhor eficiencia?
```

Isso cria um baseline real. Depois, quando formos para 70B, teremos uma curva de
extrapolacao, nao apenas uma demonstracao isolada.

## Proximo Passo Imediato

Implementar o Marco 1:

- memory planner para DRM full 125M/350M;
- estimativa de parametros das configs candidatas;
- recomendacao automatica do maior tamanho viavel na RTX 4090;
- atualizar roadmap com o tamanho escolhido.
