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
| SAINT | 3 | 6.814889 | 6.814889 | 0.00000200 |
| LoRA | 3 | 6.808302 | 6.806123 | 0.00000399 |

Pico CUDA observado:

```text
SAINT: 2.083841536 GB
LoRA:  1.016675840 GB
```

Artefato LoRA carregado:

```text
lora_loaded_validation_loss: 6.80828857421875
lora_loaded_perplexity: 905.3200922133876
```

Geracao simples:

```text
SAINT      -> SAINT-RADIO-SOUTH
Checkpoint -> Checkpoint\n\nThe following is a list of
Training   -> Training.\n\nThe first step is to
```

### Veredito

```text
nao avancar ainda para 3B
```

Motivos:

- LoRA venceu SAINT em validation loss media;
- LoRA venceu em ganho por parametro;
- LoRA usou menos pico CUDA neste caminho atual;
- SAINT ainda carrega mais estado/runtime do que deveria para esse porte;
- o caminho SAINT precisa reduzir memoria e melhorar selecao de deltas antes do
  experimento 3B.

### Problema Corrigido

O smoke em GPT-2 small revelou um bug no merge quando o adapter Hugging Face
usava `max_dim` menor que a matriz real. O delta salvo estava no shape completo
da matriz, enquanto o peso base do merge estava fatiado.

Correcao:

```text
delta_payload agora corta o delta para o mesmo shape da base antes de salvar.
```

## Marco 2 - Otimizar SAINT em GPT-2 Small

Status: **concluido**.

Este marco reduziu custo do caminho SAINT antes de tentar um modelo 3B.

### Entregas

- o treino Hugging Face reutiliza o `state_dict` do modelo ja carregado e evita
  uma segunda carga completa via `make_task`;
- checkpoints do caminho `hf_saint_forward_smoke` salvam deltas esparsos no
  formato `saint_sparse_delta`, contendo apenas valores treinaveis;
- o merge parcial usa `merge_runtime(..., matrix_names=...)`;
- o leitor de delta esparso filtra por matriz antes de expandir;
- a validacao registra memoria por etapa:
  `load_cuda_peak_bytes`, `train_cuda_peak_bytes`, `checkpoint_file_bytes` e
  `merge_cuda_peak_bytes`;
- foi testada uma curva SAINT contra LoRA rank `2` e `4`.

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
| SAINT | budget 16 | 3 | 6.814889 | 6.814889 | 0.00000200 | 2.079 |
| SAINT | budget 64 | 3 | 6.814830 | 6.814830 | 0.00000328 | 2.076 |
| SAINT | budget 256 | 3 | 6.814630 | 6.814630 | 0.00000302 | 2.076 |
| LoRA | rank 2 | 3 | 6.808302 | 6.806123 | 0.00000399 | 1.017 |
| LoRA | rank 4 | 3 | 6.799007 | 6.794116 | 0.00000418 | 1.017 |

Agregado:

```text
SAINT: mean val loss 6.814783, best 6.814630, mean gain/param 0.00000276
LoRA:  mean val loss 6.803654, best 6.794116, mean gain/param 0.00000408
```

Memoria por etapa em um run SAINT:

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

O caminho ficou mais eficiente e os checkpoints agora sao esparsos, mas SAINT
ainda perde para LoRA em GPT-2 small em loss, ganho por parametro e pico CUDA.

### Plano do Marco 3

Marco 3 deve melhorar a competitividade SAINT em GPT-2 small:

- selecionar deltas por gradiente real, nao por magnitude inicial;
- testar mais matrizes alvo por camada;
- aumentar steps e validar se SAINT ganha com treino mais longo;
- comparar budgets maiores sem aumentar payload denso;
- reduzir overhead CUDA do forward funcional;
- manter LoRA rank `2` e `4` como controle minimo.

## Marco 3 - Roteamento por Gradiente em GPT-2 Small

Status: **concluido**.

Este marco trocou a selecao SAINT de deltas por magnitude inicial para gradiente
real da loss. O caminho agora calcula um mapa de sensibilidade por autograd,
seleciona os maiores gradientes dentro do budget e treina apenas esses valores
como parametros reais do otimizador.

### Mudancas Tecnicas

- `hf_saint_forward_smoke` aceita `routing_method=gradient`;
- o delta treinavel deixou de ser uma matriz densa mascarada;
- AdamW otimiza apenas os valores selecionados;
- o payload continua esparso;
- o benchmark aceita `--saint-target-matrices`;
- o experimento usou 4 matrizes alvo SAINT.

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
| SAINT | budget 256 | 3 | 6.833445 | 6.833445 | 0.00068272 | 2.263 |
| SAINT | budget 1024 | 3 | 6.791341 | 6.791341 | 0.00022535 | 2.262 |
| SAINT | budget 4096 | 3 | 6.704383 | 6.704383 | 0.00008584 | 2.262 |
| LoRA | rank 2 | 3 | 6.771237 | 6.755111 | 0.00003135 | 1.018 |
| LoRA | rank 4 | 3 | 6.741114 | 6.727021 | 0.00001918 | 1.018 |

Agregado:

```text
SAINT: mean val loss 6.776390, best 6.704383, mean gain/param 0.00033130
LoRA:  mean val loss 6.756175, best 6.727021, mean gain/param 0.00002527
```

Memoria por etapa em um run SAINT:

```text
load_cuda_peak_bytes: 508782592
train_cuda_peak_bytes: 2263763456
checkpoint_file_bytes: 527070
merge_cuda_peak_bytes: 18087936
```

### Veredito

```text
SAINT ficou competitivo em qualidade, mas ainda nao em memoria.
```

O melhor SAINT (`budget=4096`) venceu o melhor LoRA em validation loss e o ganho
medio por parametro foi maior. A ressalva e que o pico CUDA do caminho SAINT
continua aproximadamente 2.2x maior que LoRA.

### Teste com Learning Rate 0.005

O mesmo grid foi repetido com `--saint-lrs 0.005` e `--lora-lrs 0.005`.

Resultado:

```text
SAINT: mean val loss 6.562497, best 6.140045, mean gain/param 0.00049000
LoRA:  mean val loss 6.664005, best 6.563403, mean gain/param 0.00004904
```

Curva:

| metodo | config | count | mean val loss | best val loss | mean gain/param | mean CUDA GB |
|---|---:|---:|---:|---:|---:|---:|
| SAINT | budget 256 | 3 | 6.743636 | 6.743636 | 0.00085865 | 2.263 |
| SAINT | budget 1024 | 3 | 6.803809 | 6.803809 | 0.00024298 | 2.262 |
| SAINT | budget 4096 | 3 | 6.140045 | 6.140045 | 0.00036838 | 2.262 |
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
- o payload base do checkpoint SAINT guarda apenas matrizes alvo por padrao;
- benchmarks de validacao chamam `merge_runtime(..., write_artifact=False)`;
- `merge_runtime` ganhou opcao `write_artifact`;
- as metricas separam `routing_cuda_peak_bytes` e `train_cuda_peak_bytes`.

### Resultado

Com o mesmo grid `lr=0.005`, a qualidade permaneceu igual:

```text
SAINT: mean val loss 6.562497, best 6.140045, mean gain/param 0.00049000
LoRA:  mean val loss 6.664005, best 6.563403, mean gain/param 0.00004904
```

Memoria por etapa em um run SAINT:

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

SAINT esta competitivo em qualidade no GPT-2 small, mas o roteamento por
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

| roteamento | SAINT mean val loss | SAINT best val loss | routing CUDA GB | train CUDA GB | decisao |
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

| roteamento | SAINT mean val loss | SAINT best val loss | routing CUDA GB | train CUDA GB | decisao |
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
- o pico total SAINT ainda fica acima de LoRA, mas a RTX 4090 tem margem para
  uma ponte 3B controlada se o modelo base for carregado de forma conservadora.

### Condicoes Para Ponte 3B

- usar `activation` como roteador inicial;
- manter `gradient_sequential` como controle de qualidade;
- `micro_batch=1`;
- `routing_max_length` baixo;
- salvar apenas deltas esparsos;
- nao tentar full fine-tuning;
- abortar se pico CUDA passar do budget definido.

## Proximo Marco

Marco 7 deve iniciar ponte 3B controlada:

- escolher modelo causal LM proximo de 3B que caiba localmente;
- carregar em CUDA com dtype economico quando suportado;
- rodar smoke sem treino primeiro;
- rodar SAINT `activation` com `budget=4096` ou menor;
- comparar contra LoRA pequeno se couber;
- medir load/routing/train/checkpoint/merge;
- decidir se Fase 14 fecha ou precisa de mais otimizacao.
