# Fase 14 - Escala 3B

Status: **em andamento**.

## Objetivo

Preparar a passagem de modelos pequenos para modelos de escala maior em GPU
domestica, sem pular diretamente para 3B.

## Marco 1 - Ponte GPT-2 Small

Status: **concluido**.

Este marco usa `gpt2` como ponte entre `sshleifer/tiny-gpt2` e um modelo 3B.

### Modelo

```text
modelo: gpt2
parametros: 124.439.808
local: models/gpt2
```

### Comando

```bash
python scripts/benchmark_huggingface_multiseed_phase13.py \
  --model models/gpt2 \
  --corpus data/tinyshakespeare_phase13.txt \
  --out runs/phase14_marco1_gpt2_bridge \
  --device cuda \
  --steps 2 \
  --batch-size 4 \
  --seeds 31,32,33 \
  --saint-budgets 16 \
  --saint-lrs 0.001 \
  --lora-ranks 2 \
  --lora-lrs 0.001
```

### Resultado CUDA

| metodo | count | mean val loss | best val loss | mean gain/param |
|---|---:|---:|---:|---:|
| SAINT-G | 3 | 6.814889 | 6.814889 | 0.00000200 |
| LoRA | 3 | 6.808302 | 6.806123 | 0.00000399 |

Pico CUDA observado:

```text
SAINT-G: 2.083841536 GB
LoRA:  1.016675840 GB
```

Artefato LoRA carregado:

```text
lora_loaded_validation_loss: 6.80828857421875
lora_loaded_perplexity: 905.3200922133876
```

Geracao simples:

```text
SAINT-G      -> SAINT-G-RADIO-SOUTH
Checkpoint -> Checkpoint\n\nThe following is a list of
Training   -> Training.\n\nThe first step is to
```

### Veredito

```text
nao avancar ainda para 3B
```

Motivos:

- LoRA venceu SAINT-G em validation loss media;
- LoRA venceu em ganho por parametro;
- LoRA usou menos pico CUDA neste caminho atual;
- SAINT-G ainda carrega mais estado/runtime do que deveria para esse porte;
- o caminho SAINT-G precisa reduzir memoria e melhorar selecao de deltas antes do
  experimento 3B.

### Problema Corrigido

O smoke em GPT-2 small revelou um bug no merge quando o adapter Hugging Face
usava `max_dim` menor que a matriz real. O delta salvo estava no shape completo
da matriz, enquanto o peso base do merge estava fatiado.

Correcao:

```text
delta_payload agora corta o delta para o mesmo shape da base antes de salvar.
```

## Marco 2 - Otimizar SAINT-G em GPT-2 Small

Status: **concluido**.

Este marco reduziu custo do caminho SAINT-G antes de tentar um modelo 3B.

### Entregas

- o treino Hugging Face reutiliza o `state_dict` do modelo ja carregado e evita
  uma segunda carga completa via `make_task`;
- checkpoints do caminho `hf_SAINT-G_forward_smoke` salvam deltas esparsos no
  formato `SAINT-G_sparse_delta`, contendo apenas valores treinaveis;
- o merge parcial usa `merge_runtime(..., matrix_names=...)`;
- o leitor de delta esparso filtra por matriz antes de expandir;
- a validacao registra memoria por etapa:
  `load_cuda_peak_bytes`, `train_cuda_peak_bytes`, `checkpoint_file_bytes` e
  `merge_cuda_peak_bytes`;
- foi testada uma curva SAINT-G contra LoRA rank `2` e `4`.

### Comando

```bash
python scripts/benchmark_huggingface_multiseed_phase13.py \
  --model models/gpt2 \
  --corpus data/tinyshakespeare_phase13.txt \
  --out runs/phase14_marco2_gpt2_optimized \
  --device cuda \
  --steps 2 \
  --batch-size 4 \
  --seeds 31,32,33 \
  --saint-budgets 16,64,256 \
  --saint-lrs 0.001 \
  --lora-ranks 2,4 \
  --lora-lrs 0.001
```

### Resultado CUDA

| metodo | config | count | mean val loss | best val loss | mean gain/param | mean CUDA GB |
|---|---:|---:|---:|---:|---:|---:|
| SAINT-G | budget 16 | 3 | 6.814889 | 6.814889 | 0.00000200 | 2.079 |
| SAINT-G | budget 64 | 3 | 6.814830 | 6.814830 | 0.00000328 | 2.076 |
| SAINT-G | budget 256 | 3 | 6.814630 | 6.814630 | 0.00000302 | 2.076 |
| LoRA | rank 2 | 3 | 6.808302 | 6.806123 | 0.00000399 | 1.017 |
| LoRA | rank 4 | 3 | 6.799007 | 6.794116 | 0.00000418 | 1.017 |

Agregado:

```text
SAINT-G: mean val loss 6.814783, best 6.814630, mean gain/param 0.00000276
LoRA:  mean val loss 6.803654, best 6.794116, mean gain/param 0.00000408
```

Memoria por etapa em um run SAINT-G:

```text
load_cuda_peak_bytes: 508782592
train_cuda_peak_bytes: 2045830144
checkpoint_file_bytes: 273852
merge_cuda_peak_bytes: 18087936
```

### Veredito

```text
nao avancar ainda para 3B
```

O caminho ficou mais eficiente e os checkpoints agora sao esparsos, mas SAINT-G
ainda perde para LoRA em GPT-2 small em loss, ganho por parametro e pico CUDA.

### Plano do Marco 3

Marco 3 deve melhorar a competitividade SAINT-G em GPT-2 small:

- selecionar deltas por gradiente real, nao por magnitude inicial;
- testar mais matrizes alvo por camada;
- aumentar steps e validar se SAINT-G ganha com treino mais longo;
- comparar budgets maiores sem aumentar payload denso;
- reduzir overhead CUDA do forward funcional;
- manter LoRA rank `2` e `4` como controle minimo.

## Marco 3 - Roteamento por Gradiente em GPT-2 Small

Status: **concluido**.

Este marco trocou a selecao SAINT-G de deltas por magnitude inicial para gradiente
real da loss. O caminho agora calcula um mapa de sensibilidade por autograd,
seleciona os maiores gradientes dentro do budget e treina apenas esses valores
como parametros reais do otimizador.

### Mudancas Tecnicas

- `hf_SAINT-G_forward_smoke` aceita `routing_method=gradient`;
- o delta treinavel deixou de ser uma matriz densa mascarada;
- AdamW otimiza apenas os valores selecionados;
- o payload continua esparso;
- o benchmark aceita `--saint-target-matrices`;
- o experimento usou 4 matrizes alvo SAINT-G.

### Comando

```bash
python scripts/benchmark_huggingface_multiseed_phase13.py \
  --model models/gpt2 \
  --corpus data/tinyshakespeare_phase13.txt \
  --out runs/phase14_marco3_gpt2_gradient \
  --device cuda \
  --steps 8 \
  --batch-size 4 \
  --seeds 31,32,33 \
  --saint-budgets 256,1024,4096 \
  --saint-lrs 0.001 \
  --lora-ranks 2,4 \
  --lora-lrs 0.001 \
  --saint-target-matrices 4 \
  --saint-routing-method gradient
```

### Resultado CUDA

| metodo | config | count | mean val loss | best val loss | mean gain/param | mean CUDA GB |
|---|---:|---:|---:|---:|---:|---:|
| SAINT-G | budget 256 | 3 | 6.833445 | 6.833445 | 0.00068272 | 2.263 |
| SAINT-G | budget 1024 | 3 | 6.791341 | 6.791341 | 0.00022535 | 2.262 |
| SAINT-G | budget 4096 | 3 | 6.704383 | 6.704383 | 0.00008584 | 2.262 |
| LoRA | rank 2 | 3 | 6.771237 | 6.755111 | 0.00003135 | 1.018 |
| LoRA | rank 4 | 3 | 6.741114 | 6.727021 | 0.00001918 | 1.018 |

Agregado:

```text
SAINT-G: mean val loss 6.776390, best 6.704383, mean gain/param 0.00033130
LoRA:  mean val loss 6.756175, best 6.727021, mean gain/param 0.00002527
```

Memoria por etapa em um run SAINT-G:

```text
load_cuda_peak_bytes: 508782592
train_cuda_peak_bytes: 2263763456
checkpoint_file_bytes: 527070
merge_cuda_peak_bytes: 18087936
```

### Veredito

```text
SAINT-G ficou competitivo em qualidade, mas ainda nao em memoria.
```

O melhor SAINT-G (`budget=4096`) venceu o melhor LoRA em validation loss e o ganho
medio por parametro foi maior. A ressalva e que o pico CUDA do caminho SAINT-G
continua aproximadamente 2.2x maior que LoRA.

### Teste com Learning Rate 0.005

O mesmo grid foi repetido com `--saint-lrs 0.005` e `--lora-lrs 0.005`.

Resultado:

```text
SAINT-G: mean val loss 6.562497, best 6.140045, mean gain/param 0.00049000
LoRA:  mean val loss 6.664005, best 6.563403, mean gain/param 0.00004904
```

Curva:

| metodo | config | count | mean val loss | best val loss | mean gain/param | mean CUDA GB |
|---|---:|---:|---:|---:|---:|---:|
| SAINT-G | budget 256 | 3 | 6.743636 | 6.743636 | 0.00085865 | 2.263 |
| SAINT-G | budget 1024 | 3 | 6.803809 | 6.803809 | 0.00024298 | 2.262 |
| SAINT-G | budget 4096 | 3 | 6.140045 | 6.140045 | 0.00036838 | 2.262 |
| LoRA | rank 2 | 3 | 6.642759 | 6.563403 | 0.00007042 | 1.018 |
| LoRA | rank 4 | 3 | 6.685251 | 6.633552 | 0.00002766 | 1.018 |

Decisao automatica:

```text
fase_13_can_close_with_caveat
```

## Proximo Marco

Marco 4 deve reduzir overhead de memoria antes do 3B:

- evitar `functional_call` com dicionario completo a cada step, se possivel;
- testar aplicacao temporaria dos deltas diretamente nos parametros alvo;
- medir o custo isolado do mapa de gradiente;
- reduzir matrizes carregadas no payload base para apenas alvos treinaveis;
- repetir GPT-2 small com `budget=4096` e steps maiores;
- so iniciar 3B se o pico CUDA cair para perto de LoRA ou se houver margem clara
  na RTX 4090.

## Marco 4 - Diagnostico de Memoria CUDA

Status: **concluido com ressalvas**.

### Mudancas

- `functional_call` agora recebe apenas parametros alterados, em vez de um
  dicionario completo de parametros;
- o payload base do checkpoint SAINT-G guarda apenas matrizes alvo por padrao;
- benchmarks de validacao chamam `merge_runtime(..., write_artifact=False)`;
- `merge_runtime` ganhou opcao `write_artifact`;
- as metricas separam `routing_cuda_peak_bytes` e `train_cuda_peak_bytes`.

### Resultado

Com o mesmo grid `lr=0.005`, a qualidade permaneceu igual:

```text
SAINT-G: mean val loss 6.562497, best 6.140045, mean gain/param 0.00049000
LoRA:  mean val loss 6.664005, best 6.563403, mean gain/param 0.00004904
```

Memoria por etapa em um run SAINT-G:

```text
load_cuda_peak_bytes: 508782592
routing_cuda_peak_bytes: 2263763456
train_cuda_peak_bytes: 633309184
checkpoint_file_bytes: 19342
merge_cuda_peak_bytes: 18087936
```

Comparacao com Marco 3:

```text
checkpoint/artifact caiu de ~527 KB para ~19 KB
train isolado ficou em ~0.64 GB
pico total ainda e dominado pelo roteamento por gradiente: ~2.26 GB
```

### Veredito

SAINT-G esta competitivo em qualidade no GPT-2 small, mas o roteamento por
gradiente precisa ficar mais barato antes de um salto confiante para 3B.

## Proximo Marco

Marco 5 deve reduzir memoria do roteamento:

- calcular gradiente apenas por matriz alvo, uma matriz por vez;
- limpar cache entre matrizes alvo;
- escolher top-k global a partir de scores em CPU;
- testar `torch.no_grad()`/descarte agressivo para tensors intermediarios;
- comparar roteamento por gradiente completo contra roteamento aproximado mais
  barato;
- repetir `budget=4096`, `lr=0.005` antes de decidir ponte 3B.

## Marco 5 - Roteamento Sequencial e Scores em CPU

Status: **concluido com ressalvas**.

### Mudancas

- `routing_method=gradient_sequential` calcula sensibilidade uma matriz alvo por
  vez;
- cada score de gradiente e movido para CPU antes do `top-k` global;
- o cache CUDA e limpo entre matrizes alvo;
- `routing_method=magnitude` foi usado como controle barato aproximado;
- o benchmark foi repetido com `budget=4096`, `lr=0.005`.

### Comparacao

| roteamento | SAINT-G mean val loss | SAINT-G best val loss | routing CUDA GB | train CUDA GB | decisao |
|---|---:|---:|---:|---:|---|
| gradient completo | 6.562497 | 6.140045 | 2.264 | 0.638 | passa com ressalva |
| gradient sequencial | 6.140045 | 6.140045 | 2.247 | 0.637 | passa com ressalva |
| magnitude | 6.751935 | 6.751935 | 0.518 | 0.637 | falha contra LoRA |

Controle LoRA no mesmo grid:

```text
LoRA: mean val loss 6.664005, best 6.563403, mean gain/param 0.00004904
```

Memoria por etapa do `gradient_sequential`:

```text
load_cuda_peak_bytes: 508782592
routing_cuda_peak_bytes: 2246670848
train_cuda_peak_bytes: 633692672
checkpoint_file_bytes: 19605
merge_cuda_peak_bytes: 18087936
```

Memoria por etapa do roteador `magnitude`:

```text
load_cuda_peak_bytes: 508782592
routing_cuda_peak_bytes: 518262784
train_cuda_peak_bytes: 633430528
checkpoint_file_bytes: 19572
merge_cuda_peak_bytes: 18087936
```

### Veredito

O roteamento sequencial preservou a qualidade do gradiente completo, mas reduziu
pouco a memoria. Isso indica que o custo dominante nao e acumular gradientes de
varias matrizes ao mesmo tempo, e sim o proprio backward usado para medir
sensibilidade.

O roteamento por magnitude mostrou o piso de memoria desejado, mas perdeu para
LoRA. Portanto, a proxima melhoria deve aproximar o beneficio do gradiente sem
rodar um backward completo caro.

## Proximo Marco

Marco 6 deve testar roteamento aproximado de baixo custo:

- usar gradiente da ultima camada ou `lm_head` como proxy;
- testar sensibilidade por ativacao sem backward completo;
- testar score hibrido `magnitude * ativacao`;
- testar subset de batch/seq_len menor apenas para roteamento;
- comparar qualidade/memoria contra `gradient_sequential`;
- decidir se GPT-2 small ja autoriza ponte 3B com ressalva ou se falta outro
  ciclo de roteamento.

## Marco 6 - Roteamento Aproximado de Baixo Custo

Status: **concluido**.

### Mudancas

- roteamento foi separado em `saint/adapters/huggingface_routing.py`;
- `routing_method=activation` usa ativacoes capturadas por hooks, sem backward;
- `routing_method=magnitude_activation` usa `magnitude * ativacao`;
- `routing_method=lm_head_proxy` usa proxy barato combinado com `lm_head`;
- o benchmark aceita `--saint-routing-max-length` e
  `--saint-routing-batch-size`;
- o roteamento pode usar subset menor que o treino.

### Comando Base

```bash
python scripts/benchmark_huggingface_multiseed_phase13.py \
  --model models/gpt2 \
  --corpus data/tinyshakespeare_phase13.txt \
  --device cuda \
  --steps 8 \
  --batch-size 4 \
  --seeds 31,32,33 \
  --saint-budgets 4096 \
  --saint-lrs 0.005 \
  --lora-ranks 2,4 \
  --lora-lrs 0.005 \
  --saint-target-matrices 4 \
  --saint-routing-max-length 8 \
  --saint-routing-batch-size 1
```

### Comparacao

| roteamento | SAINT-G mean val loss | SAINT-G best val loss | routing CUDA GB | train CUDA GB | decisao |
|---|---:|---:|---:|---:|---|
| gradient sequencial completo | 6.140045 | 6.140045 | 2.259 | 0.637 | passa com ressalva |
| gradient sequencial subset | 6.441031 | 6.441031 | 0.540 | 0.637 | passa com ressalva |
| activation subset | 6.522398 | 6.522398 | 0.519 | 0.613 | passa com ressalva |
| magnitude activation subset | 6.716236 | 6.716236 | 0.528 | 0.637 | falha contra LoRA |
| lm_head proxy subset | 6.716236 | 6.716236 | 0.528 | 0.637 | falha contra LoRA |

Controle LoRA:

```text
LoRA: mean val loss 6.664005, best 6.563403, mean gain/param 0.00004904
```

### Veredito

```text
GPT-2 small autoriza uma ponte 3B experimental com ressalva.
```

Motivo:

- `activation` nao usa backward para sensibilidade;
- reduziu roteamento de ~2.26 GB para ~0.52 GB;
- venceu LoRA em media de validation loss neste grid;
- `gradient_sequential` com subset tambem venceu LoRA e ficou barato;
- o pico total SAINT-G ainda fica acima de LoRA, mas a RTX 4090 tem margem para
  uma ponte 3B controlada se o modelo base for carregado de forma conservadora.

### Condicoes Para Ponte 3B

- usar `activation` como roteador inicial;
- manter `gradient_sequential` como controle de qualidade;
- `micro_batch=1`;
- `routing_max_length` baixo;
- salvar apenas deltas esparsos;
- nao tentar full fine-tuning;
- abortar se pico CUDA passar do budget definido.

## Marco 7 - Ponte 3B Controlada

Status: **concluido com ressalva**.

### Modelo Escolhido

Modelo local:

```text
Qwen/Qwen2.5-3B
```

Motivo:

- causal LM base;
- proximo de 3B parametros;
- checkpoint local de aproximadamente 6.18 GB;
- carrega em CUDA com `bfloat16` dentro do budget da RTX 4090.

### Mudancas

- benchmarks HF aceitam `--model-dtype` e `--max-cuda-gb`;
- `from_pretrained` usa dtype economico quando configurado;
- caminho SAINT-G aborta se `load`, `routing` ou `train` excederem budget CUDA;
- CLI aceita `--skip-lora` e `--skip-generation` para smoke controlado;
- alvos default agora cobrem familias GPT e Qwen:
  `q_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`;
- deltas SAINT-G usam o mesmo dtype do peso alvo;
- roteamento pode limitar scores ao recorte salvo no checkpoint, evitando delta
  treinado fora do payload;
- LoRA bf16 faz merge do delta no dtype do peso base.

### Smoke Sem Treino

```text
model_dtype: float16
load_s: 14.986
forward_s: 0.284
load_cuda_gb: 5.854
peak_cuda_gb: 5.873
loss: 8.064380
```

O smoke passou dentro do budget.

### SAINT-G Activation

Configuracao:

```text
model_dtype: bfloat16
budget: 4096
steps: 1
batch_size: 1
routing_method: activation
routing_max_length: 8
routing_batch_size: 1
max_cuda_gb: 22
```

Resultado:

| metrica | valor |
|---|---:|
| base validation loss | 7.690704 |
| SAINT-G validation loss | 7.688824 |
| merged validation loss | 7.688824 |
| parametros treinaveis | 4096 |
| delta sparse values | 4096 |
| delta only bytes | 113307 |
| checkpoint bytes | 309252 |
| load CUDA GB | 5.863 |
| routing CUDA GB | 5.864 |
| train CUDA GB | 6.016 |
| peak CUDA GB | 11.869 |
| merge CUDA GB | 0.017 |
| tokens/s | 97.155 |
| resume quality delta | 0.000000 |

### Comparacao LoRA Minima

Controle:

```text
LoRA rank 1
learning_rate: 0.005
steps: 1
batch_size: 1
model_dtype: bfloat16
```

| metodo | val loss | params | gain/param | CUDA peak GB |
|---|---:|---:|---:|---:|
| SAINT-G activation | 7.688824 | 4096 | 0.00000613 | 11.869 |
| LoRA rank 1 | 7.664804 | 6400 | 0.00000684 | 7.630 |

LoRA rank 1 coube e venceu neste microteste.

### Veredito

```text
Fase 14 Marco 7 passou tecnicamente, mas Fase 14 ainda nao fecha.
```

O projeto agora consegue:

- carregar um modelo 3B local em CUDA;
- rodar smoke sem treino;
- treinar SAINT-G activation com micro-batch 1;
- salvar delta esparso real;
- validar resume/merge;
- comparar contra LoRA pequeno.

Mas SAINT-G ainda nao esta competitivo contra LoRA rank 1 em qualidade e pico CUDA
neste regime. A proxima etapa deve reduzir overhead de memoria e melhorar o
ganho por parametro antes de declarar a Fase 14 concluida.

## Marco 8 - Otimizacao SAINT-G 3B contra LoRA

Status: **concluido com ressalva**.

### Mudancas

- o caminho `hf_SAINT-G_forward_smoke` deixou de materializar recortes densos via
  `matrices_from_state` durante o treino real;
- o payload `SAINT-G_sparse_delta` agora salva shapes completos e coordenadas
  reais dos tensores alvo;
- checkpoints esparsos podem ser validados sem expandir para matriz densa;
- benchmarks HF leem e aplicam deltas esparsos diretamente por coordenada;
- `merge_runtime` continua compativel com deltas esparsos, recortando para o
  shape base quando necessario;
- `bfloat16` foi adotado como dtype operacional para os testes 3B.

### Grid Activation

Configuracao:

```text
model: Qwen/Qwen2.5-3B
dtype: bfloat16
seeds: 31, 32
budgets SAINT-G: 8192, 16384
LoRA ranks: 1, 2
steps: 1
batch_size: 1
routing_method: activation
routing_max_length: 8
routing_batch_size: 1
max_cuda_gb: 22
```

Agregado:

| metodo | count | mean val loss | best val loss | mean gain/param |
|---|---:|---:|---:|---:|
| SAINT-G activation | 4 | 7.657179 | 7.656769 | 0.00000225 |
| LoRA rank 1/2 | 4 | 7.671440 | 7.661643 | 0.00000430 |

Por linha:

| metodo | seed | budget/rank | val loss | params | peak CUDA GB | merge CUDA GB |
|---|---:|---:|---:|---:|---:|---:|
| SAINT-G activation | 31 | 8192 | 7.657590 | 8192 | 11.827 | 8.094 |
| SAINT-G activation | 31 | 16384 | 7.656769 | 16384 | 11.827 | 8.094 |
| LoRA | 31 | rank 1 | 7.664804 | 6400 | 7.630 | n/a |
| LoRA | 31 | rank 2 | 7.661643 | 12800 | 7.630 | n/a |
| SAINT-G activation | 32 | 8192 | 7.657590 | 8192 | 11.827 | 8.094 |
| SAINT-G activation | 32 | 16384 | 7.656769 | 16384 | 11.827 | 8.094 |
| LoRA | 32 | rank 1 | 7.678699 | 6400 | 7.630 | n/a |
| LoRA | 32 | rank 2 | 7.680615 | 12800 | 7.630 | n/a |

### Controle Gradient Sequential Subset

Configuracao:

```text
seed: 31
budgets: 8192, 16384
routing_method: gradient_sequential
routing_max_length: 8
routing_batch_size: 1
dtype: bfloat16
max_cuda_gb: 22
```

Resultado:

| metodo | budget | val loss | params | load CUDA GB | routing CUDA GB | train CUDA GB | merge CUDA GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| gradient_sequential | 8192 | 7.460260 | 8192 | 5.863 | 5.956 | 5.969 | 8.084 |
| gradient_sequential | 16384 | 7.448014 | 16384 | 5.870 | 5.956 | 5.970 | 8.084 |

O `gradient_sequential` subset melhorou qualidade sem estourar o limite de
22 GB. Ele venceu tanto o melhor SAINT-G activation quanto o melhor LoRA rank 2
neste smoke.

### Veredito

```text
Fase 14 Marco 8 passou, mas Fase 14 ainda nao fecha.
```

Motivo:

- SAINT-G voltou a vencer LoRA em validation loss no grid activation;
- `gradient_sequential` subset deu o melhor resultado do marco;
- o delta esparso agora nao depende mais de recorte denso para ser salvo/aplicado;
- o pico do forward funcional SAINT-G ainda fica maior que LoRA;
- o merge/eval ainda carrega o modelo para avaliar o delta aplicado, gerando pico
  CUDA relevante.

## Marco 9 - Reducao do Pico Funcional

Status: **concluido**.

### Mudancas

- adicionado `--saint-delta-application`;
- `functional` preserva o caminho anterior com `functional_call`;
- `inplace` aplica o delta no peso real antes do forward, faz rollback depois do
  backward e atualiza apenas os valores esparsos selecionados;
- memoria de merge/eval agora separa:
  - pico de load do modelo;
  - pico do forward com delta aplicado;
- o caminho in-place evita `zeros_like` completo por loss, mas usa gradiente do
  peso alvo para atualizar as coordenadas esparsas.

### Functional vs Inplace

Configuracao:

```text
model: Qwen/Qwen2.5-3B
seed: 31
budget: 16384
routing_method: gradient_sequential
routing_max_length: 8
routing_batch_size: 1
dtype: bfloat16
max_cuda_gb: 22
```

Resultado:

| delta application | val loss | train loss | peak CUDA GB | train CUDA GB | merge load GB | merge eval GB | tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| functional | 7.448014 | 7.389300 | 11.785 | 5.970 | 5.870 | 8.084 | 97.337 |
| inplace | 7.687493 | 7.981713 | 7.660 | 5.970 | 5.870 | 8.059 | 97.759 |

Conclusao:

- o modo `inplace` reduziu o pico total do caminho SAINT-G;
- a qualidade caiu muito porque o update virou um SGD esparso manual;
- o modo `functional` continua sendo o baseline de qualidade para 3B;
- o modo `inplace` fica como caminho experimental para uma futura versao com
  otimizador esparso melhor.

### Repeticao Multiseed

Configuracao:

```text
seeds: 31, 32, 33
budget: 16384
routing_method: gradient_sequential
delta_application: functional
dtype: bfloat16
```

Resultado:

| seed | val loss | train loss | peak CUDA GB | routing CUDA GB | train CUDA GB | merge load GB | merge eval GB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 31 | 7.448014 | 7.389300 | 11.785 | 5.956 | 5.970 | 5.870 | 8.084 |
| 32 | 7.448014 | 7.389300 | 11.785 | 5.956 | 5.970 | 5.870 | 8.084 |
| 33 | 7.448014 | 7.389300 | 11.785 | 5.956 | 5.970 | 5.870 | 8.084 |

Agregado:

| metodo | count | mean val loss | best val loss | mean gain/param |
|---|---:|---:|---:|---:|
| SAINT-G gradient_sequential functional | 3 | 7.448014 | 7.448014 | 0.00003785 |

### Decisao da Fase 14

```text
Fase 14 conclui com ressalva.
```

SAINT-G 3B cumpriu o criterio minimo:

- carrega modelo 3B local em CUDA;
- treina delta esparso com micro-batch 1;
- usa `bfloat16` de forma estavel;
- salva checkpoint esparso por coordenada;
- valida resume e merge/eval;
- vence LoRA rank 1/2 em validation loss no grid do Marco 8;
- `gradient_sequential` subset venceu com margem maior e repetiu em seeds
  31, 32 e 33.

Ressalva:

- LoRA ainda tem menor pico CUDA;
- `functional_call` ainda materializa tensores densos temporarios;
- o caminho `inplace` reduz memoria, mas ainda nao preserva qualidade.

## Proxima Fase

Fase 15 deve iniciar escala 14B com cautela:

- manter `gradient_sequential` subset funcional como baseline SAINT-G 3B;
- investigar otimizador esparso para o caminho `inplace`;
- adicionar offload/CPU para load e merge/eval;
- evitar materializacao densa no forward;
- comparar contra LoRA pequeno antes de qualquer salto para 70B.
