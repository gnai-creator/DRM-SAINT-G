# Fase 15 - Escala 14B

Status: **concluida com ressalvas**.

## Objetivo

Testar gargalos de offload, roteamento e latencia antes de qualquer salto para
70B.

Baseline herdado da Fase 14:

```text
Qwen/Qwen2.5-3B
gradient_sequential subset
budget 16384
delta_application functional
bfloat16
micro-batch 1
```

## Marco 1 - Ponte 14B Controlada

Status: **concluido com bloqueio de treino**.

### Modelo

Modelo escolhido:

```text
Qwen/Qwen2.5-14B
```

Motivo:

- causal LM base;
- mesma familia usada na ponte 3B;
- 14.7B parametros no model card;
- 48 camadas;
- tensor type BF16.

Checkpoint local:

```text
models/Qwen2.5-14B
tamanho local: 29.55 GB
```

### Mudancas

- dependencia `accelerate>=1.12` adicionada ao ambiente HF;
- helper `saint.adapters.huggingface_loading` centraliza:
  - dtype;
  - `device_map`;
  - `max_memory`;
  - `offload_folder`;
  - `low_cpu_mem_usage`;
- benchmark HF aceita:
  - `--hf-device-map`;
  - `--hf-max-memory`;
  - `--hf-offload-folder`;
- script `scripts/benchmark_huggingface_phase15_14b.py` mede load/forward
  controlado sem treino.

### Smoke 14B

Comando:

```bash
python scripts/benchmark_huggingface_phase15_14b.py \
  --model models/Qwen2.5-14B \
  --out runs/phase15_marco1_qwen25_14b_smoke.json \
  --device cuda \
  --model-dtype bfloat16 \
  --hf-device-map auto \
  --hf-max-memory "0=20GiB,cpu=64GiB" \
  --hf-offload-folder runs/phase15_marco1_offload \
  --max-length 8
```

Resultado:

| metrica | valor |
|---|---:|
| load_s | 26.964 |
| forward_s | 1.653 |
| loss | 7.261498 |
| load CUDA GB | 18.634 |
| forward CUDA GB | 19.834 |

Mapa de dispositivo:

```text
GPU: embeddings + camadas 0 a 32
CPU: camadas 33 a 47 + norm + rotary_emb + lm_head
```

### Tentativa de Treino DRM-SAINT-G

Configuracao:

```text
delta_application: inplace
routing_method: activation
budget: 4096
steps: 1
batch_size: 1
routing_max_length: 4
model_dtype: bfloat16
device_map: auto
max_memory: 0=20GiB,cpu=64GiB
max_cuda_gb: 23
```

Resultado:

```text
timeout apos 20 minutos
```

Conclusao:

- load e forward 14B sao viaveis com offload CPU;
- treino DRM-SAINT-G 14B ainda nao e viavel neste caminho;
- a latencia do offload CPU bloqueia o ciclo treino -> checkpoint -> merge;
- LoRA 14B tambem nao deve ser usado como baseline ate o caminho de treino
  parcial ficar mais barato.

### Veredito

```text
Fase 15 Marco 1 passou para smoke, mas falhou para treino.
```

O projeto demonstrou que consegue carregar e executar forward em 14B com
memoria controlada. Ainda nao demonstrou treino estavel, ganho mensuravel ou
tempo aceitavel em 14B.

## Proximo Marco

Marco 2 deve reduzir o custo antes de tentar treino novamente:

- criar roteamento por uma unica matriz alvo em camada residente na GPU;
- permitir selecionar explicitamente camadas baixas, por exemplo layer 0 ou 1;
- evitar tocar camadas offloadadas durante treino;
- testar delta somente em `model.layers.0.self_attn.q_proj.weight`;
- medir forward com `max_memory` menor e maior;
- comparar smoke 14B contra Qwen2.5-3B para custo relativo;
- so tentar LoRA 14B depois que DRM-SAINT-G fizer um step completo abaixo de 5 minutos.

## Marco 2 - Alvo Unico em Camada Residente

Status: **concluido como diagnostico, ainda sem treino viavel**.

### Mudancas

- `saint.adapters.huggingface_forward` aceita `target_names` explicito;
- o runtime DRM-SAINT-G aceita filtrar matriz alvo por `target_device`;
- o benchmark multiseed aceita:
  - `--saint-target-names`;
  - `--saint-target-device`;
- novo probe `scripts/benchmark_huggingface_phase15_target_probe.py` lista
  dispositivo, shape e tamanho das matrizes alvo.

### Probe de Alvos

Comando:

```bash
python scripts/benchmark_huggingface_phase15_target_probe.py \
  --model models/Qwen2.5-14B \
  --out runs/phase15_marco2_target_probe_14b.json \
  --device cuda \
  --model-dtype bfloat16 \
  --hf-device-map auto \
  --hf-max-memory "0=20GiB,cpu=64GiB" \
  --hf-offload-folder runs/phase15_marco2_offload \
  --target-names model.layers.0.self_attn.q_proj.weight,model.layers.33.self_attn.q_proj.weight
```

Resultado:

| matriz | device | shape | numel |
|---|---|---:|---:|
| `model.layers.0.self_attn.q_proj.weight` | `cuda:0` | 5120x5120 | 26214400 |
| `model.layers.33.self_attn.q_proj.weight` | `meta`/CPU offload | 5120x5120 | 26214400 |

Conclusao:

```text
camadas baixas podem ser escolhidas como alvo residente em GPU;
camadas altas ficam offloadadas e nao devem ser usadas no primeiro treino 14B.
```

### Smoke por Memoria

| modelo/config | load_s | forward_s | load CUDA GB | forward CUDA GB |
|---|---:|---:|---:|---:|
| Qwen2.5-3B sem offload | 15.242 | 0.339 | 5.854 | 5.874 |
| Qwen2.5-14B `0=18GiB,cpu=64GiB` | 33.053 | 1.353 | 16.579 | 17.783 |
| Qwen2.5-14B `0=20GiB,cpu=64GiB` | 26.964 | 1.653 | 18.634 | 19.834 |
| Qwen2.5-14B `0=22GiB,cpu=64GiB` | 23.044 | 1.098 | 20.685 | 21.885 |

Observacao:

```text
mais VRAM para o device_map reduz latencia de forward, mas aproxima o pico do
limite pratico da RTX 4090.
```

### Tentativa DRM-SAINT-G Limitada

Configuracao:

```text
target_names: model.layers.0.self_attn.q_proj.weight
target_device: cuda
delta_application: inplace
routing_method: activation
budget: 1024
steps: 1
batch_size: 1
routing_max_length: 4
max_memory: 0=18GiB,cpu=64GiB
max_cuda_gb: 23
```

Resultado:

```text
RuntimeError: CUDA budget exceeded during train: 29.856 GB
```

Leitura:

- a selecao por matriz alvo funcionou;
- a camada 0 `q_proj` estava residente em GPU;
- o caminho de validacao/treino ainda cria pico alto demais antes do step;
- LoRA 14B continua adiado, porque DRM-SAINT-G ainda nao completou um step abaixo de
  5 minutos.

### Veredito

```text
Marco 2 reduziu o escopo corretamente, mas 14B ainda nao esta pronto para treino.
```

O bloqueio agora e mais especifico: nao e escolher a matriz errada, e sim o
pico de memoria/custo do caminho de treino HF com offload.

## Proximo Marco

Marco 3 deve atacar o pico de memoria antes de tentar qualidade:

- evitar segunda carga completa no grid/validacao 14B;
- permitir modo `train-only` sem base eval, merge eval e generation;
- fazer checkpoint do delta sem recarregar o modelo;
- separar script 14B de treino minimo do grid multiseed;
- testar backward somente com uma janela curta e sem avaliacao final;
- medir memoria em subetapas: load, routing, train, checkpoint;
- so reativar LoRA 14B depois de um step DRM-SAINT-G completo abaixo de 5 minutos.

## Marco 3 - Train-Only 14B Sem Grid

Status: **concluido**.

### Mudancas

- criado `scripts/benchmark_huggingface_phase15_train_only.py`;
- o caminho HF aceita `train_only` para pular:
  - base eval;
  - merge eval;
  - generation;
  - validacao final separada;
- o adapter registra tempos de subetapas:
  - load;
  - routing;
  - train;
  - checkpoint payload;
- o adapter ativa `gradient_checkpointing` quando solicitado;
- o treino muda o modelo para `train()` durante o backward e volta para
  `eval()` depois;
- o checkpoint e escrito pelo runtime sem recarregar o modelo.

### Tentativa com 18 GiB

Configuracao:

```text
model: models/Qwen2.5-14B
target: model.layers.0.self_attn.q_proj.weight
device_map: auto
max_memory: 0=18GiB,cpu=64GiB
budget: 1024
steps: 1
max_length: 4
gradient_checkpointing: true
train_only: true
```

Resultado:

```text
RuntimeError: CUDA budget exceeded during train: 29.647 GB
```

Leitura:

```text
18 GiB ainda deixa camadas demais em GPU; o backward com offload cria pico acima
do limite da RTX 4090.
```

### Tentativa com 14 GiB

Comando:

```bash
python scripts/benchmark_huggingface_phase15_train_only.py \
  --model models/Qwen2.5-14B \
  --out runs/phase15_marco3_qwen25_14b_train_only_layer0_14gib \
  --device cuda \
  --model-dtype bfloat16 \
  --hf-device-map auto \
  --hf-max-memory "0=14GiB,cpu=64GiB" \
  --hf-offload-folder runs/phase15_marco3_offload_train_only_14gib \
  --target-names model.layers.0.self_attn.q_proj.weight \
  --target-device cuda \
  --budget 1024 \
  --steps 1 \
  --batch-size 1 \
  --train-texts 1 \
  --max-length 4 \
  --routing-method activation \
  --routing-max-length 4 \
  --routing-batch-size 1 \
  --learning-rate 0.001 \
  --max-cuda-gb 23 \
  --gradient-checkpointing
```

Resultado:

| metrica | valor |
|---|---:|
| status | ok |
| elapsed_s_total | 45.948 |
| train_s | 3.444 |
| routing_s | 2.219 |
| load_s | 37.341 |
| checkpoint_payload_s | 0.000463 |
| train_loss | 6.944987 |
| tokens/s | 0.871 |
| parametros treinaveis | 1024 |
| delta JSON bytes | 81818 |
| optimizer bytes | 163 |
| load CUDA GB | 13.402 |
| routing CUDA GB | 14.687 |
| train CUDA GB | 17.401 |

Checkpoint:

```text
runs/phase15_marco3_qwen25_14b_train_only_layer0_14gib
```

Arquivos:

```text
deltas.DRM-SAINT-Gdelta.json
optimizer.DRM-SAINT-Gopt
checkpoint.json
metrics.json
logs.jsonl
```

### Veredito

```text
Marco 3 passou.
```

DRM-SAINT-G completou um step autograd 14B em modo train-only abaixo de 5 minutos,
com pico de treino abaixo de 23 GB e checkpoint gerado sem merge/eval.

Isto ainda nao prova qualidade. Prova que o caminho minimo:

```text
load -> routing -> train -> checkpoint
```

funciona em 14B com offload agressivo.

## Proximo Marco

Marco 4 deve transformar o smoke em experimento comparavel:

- repetir o train-only com budgets 1024, 4096 e 8192;
- testar `max_memory` 12GiB, 14GiB e 16GiB;
- medir custo de load separado do custo de treino em runs repetidos;
- adicionar avaliacao posterior opcional em processo separado;
- testar LoRA 14B rank 1 somente se couber no mesmo limite CUDA;
- comparar ganho por parametro treinavel contra o primeiro baseline LoRA viavel;
- manter o criterio de abortar se `train_cuda_peak_bytes` passar de 23 GB.

## Marco 4 - Comparacao Train-Only 14B

Status: **concluido com ressalva na avaliacao posterior**.

### Mudancas

- criado `scripts/benchmark_huggingface_phase15_compare.py`;
- cada ponto DRM-SAINT-G roda em subprocesso separado para evitar residuos de CUDA;
- o comparador gera:
  - `phase15_compare_results.json`;
  - `phase15_compare_results.md`;
- criado `scripts/benchmark_huggingface_phase15_eval_checkpoint.py`;
- a avaliacao posterior roda em processo separado a partir do checkpoint DRM-SAINT-G;
- LoRA rank 1 foi testado somente depois de DRM-SAINT-G passar no limite de CUDA.

### Resultado DRM-SAINT-G

Modelo:

```text
models/Qwen2.5-14B
```

Config comum:

```text
target: model.layers.0.self_attn.q_proj.weight
steps: 1
batch_size: 1
max_length: 4
routing_method: activation
gradient_checkpointing: true
max_cuda_gb: 23
```

| max_memory | budget | status | load_s | routing_s | train_s | train CUDA GB | checkpoint bytes |
|---|---:|---|---:|---:|---:|---:|---:|
| `0=12GiB,cpu=64GiB` | 1024 | ok | 41.072 | 3.256 | 3.818 | 15.769 | 81981 |
| `0=12GiB,cpu=64GiB` | 4096 | ok | 40.330 | 3.244 | 3.794 | 15.769 | 330309 |
| `0=12GiB,cpu=64GiB` | 8192 | ok | 39.971 | 3.298 | 3.774 | 15.769 | 657271 |
| `0=14GiB,cpu=64GiB` | 1024 | ok | 37.830 | 2.953 | 3.691 | 17.401 | 81981 |
| `0=14GiB,cpu=64GiB` | 4096 | ok | 37.306 | 2.928 | 3.631 | 17.401 | 330309 |
| `0=14GiB,cpu=64GiB` | 8192 | ok | 37.163 | 2.850 | 3.486 | 17.401 | 657271 |
| `0=16GiB,cpu=64GiB` | 1024 | ok | 33.266 | 2.256 | 3.004 | 19.032 | 81981 |
| `0=16GiB,cpu=64GiB` | 4096 | ok | 33.534 | 2.527 | 3.017 | 19.032 | 330309 |
| `0=16GiB,cpu=64GiB` | 8192 | ok | 33.494 | 2.492 | 3.052 | 19.033 | 657271 |

Leitura:

- todos os budgets testados passaram abaixo de 23 GB;
- aumentar `max_memory` reduz load/routing/train, mas aumenta pico de CUDA;
- o tamanho do checkpoint cresce aproximadamente linear com o budget;
- o custo dominante ainda e load/offload, nao o step de treino.

### LoRA Rank 1

Configuracao:

```text
rank: 1
target: model.layers.0.self_attn.q_proj.weight
max_memory: 0=14GiB,cpu=64GiB
gradient_checkpointing: true
```

Resultado:

| metrica | valor |
|---|---:|
| status | ok |
| train_s | 4.729 |
| train CUDA GB | 17.453 |
| parametros treinaveis | 10240 |
| train_loss | 9.438548 |

Comparacao direta:

```text
DRM-SAINT-G budget 8192 @ 14GiB: train_s 3.486, train CUDA 17.401 GB
LoRA rank 1 @ 14GiB:       train_s 4.729, train CUDA 17.453 GB
```

Ressalva:

```text
o Marco 4 ainda nao mede ganho real de qualidade, porque o treino usa apenas 1
step e uma janela curta. O ganho por parametro fica registrado como 0.0 neste
smoke.
```

### Avaliacao Posterior Separada

Checkpoint avaliado:

```text
runs/phase15_marco4_qwen25_14b_compare_isolated/DRM-SAINT-G_b8192_0_16GiB_cpu_64GiB
```

Resultado:

| metrica | valor |
|---|---:|
| merged_validation_loss | 6.801508 |
| merged_perplexity | 899.203 |
| merge load CUDA GB | 15.604 |
| merge eval CUDA GB | 29.682 |

Conclusao:

```text
a avaliacao posterior funciona, mas ainda nao respeita o limite de 23 GB.
```

### Veredito

```text
Marco 4 passou para treino comparavel train-only.
```

DRM-SAINT-G agora tem uma curva inicial de budgets em 14B e um primeiro baseline LoRA
rank 1 no mesmo alvo. A avaliacao posterior existe, mas precisa reduzir pico
antes de virar criterio de fechamento.

## Proximo Marco

Marco 5 deve melhorar qualidade e avaliacao:

- reduzir pico da avaliacao posterior abaixo de 23 GB;
- medir loss antes/depois no mesmo processo train-only sem duplicar memoria;
- repetir com mais steps, por exemplo 4 e 8;
- testar janela maior, por exemplo `max_length` 8 e 16;
- comparar DRM-SAINT-G budget 8192/16384 contra LoRA rank 1 em loss delta;
- salvar ganho por parametro treinavel real;
- testar target em `v_proj` e `o_proj`, nao apenas `q_proj`.

## Marco 5 - Qualidade e Avaliacao 14B

Status: **concluido**.

### Mudancas

- o modo `train_only` agora pode medir loss antes/depois no mesmo processo com
  `--measure-loss`;
- `initial_loss`, `loss_delta` e `gain_per_parameter` sao registrados para
  DRM-SAINT-G e LoRA;
- a avaliacao posterior usa `torch.no_grad()` para evitar grafo de autograd;
- LoRA rank 1 usa o mesmo texto de treino do corpus, em vez de texto fixo;
- foram testados targets:
  - `q_proj`;
  - `v_proj`;
  - `o_proj`.

### Avaliacao Posterior

Antes, a avaliacao posterior com `0=16GiB,cpu=64GiB` chegava a:

```text
merge eval CUDA: 29.682 GB
```

Depois de `no_grad` e `0=12GiB,cpu=64GiB`:

| checkpoint | validation loss | perplexity | load CUDA GB | eval CUDA GB |
|---|---:|---:|---:|---:|
| Marco 4 `q_proj` budget 8192 | 6.801508 | 899.203 | 11.199 | 12.487 |
| Marco 5 `v_proj` budget 16384 | 5.612469 | 273.819 | 11.199 | 12.487 |

Conclusao:

```text
o pico de avaliacao posterior caiu para abaixo de 23 GB.
```

### Qualidade Train-Only

Config comum:

```text
model: models/Qwen2.5-14B
max_memory: 0=12GiB,cpu=64GiB
steps: 4
max_length: 8
learning_rate: 0.005
routing_method: activation
gradient_checkpointing: true
```

| target | budget | initial loss | final loss | loss delta | ganho/param | train CUDA GB |
|---|---:|---:|---:|---:|---:|---:|
| `q_proj` | 8192 | 6.944987 | 6.923183 | -0.021804 | 2.6616e-06 | 15.821 |
| `q_proj` | 16384 | 6.944987 | 6.932558 | -0.012429 | 7.5862e-07 | 15.821 |
| `v_proj` | 8192 | 6.944987 | 6.646648 | -0.298339 | 3.6418e-05 | 15.779 |
| `v_proj` | 16384 | 6.944987 | 6.551485 | -0.393503 | 2.4017e-05 | 15.779 |
| `o_proj` | 8192 | 6.944987 | 6.889593 | -0.055394 | 6.7620e-06 | 15.821 |

Leitura:

- `v_proj` foi claramente o melhor alvo nesta tarefa curta;
- budget 16384 melhorou mais loss absoluta em `v_proj`;
- budget 8192 teve melhor ganho por parametro que 16384;
- todos os pontos ficaram abaixo de 23 GB.

### Steps e Janela

Testes adicionais em `q_proj`, budget 8192:

| variacao | final loss | loss delta | train_s | train CUDA GB |
|---|---:|---:|---:|---:|
| steps 4, max_length 8 | 6.923183 | -0.021804 | 15.043 | 15.821 |
| steps 4, max_length 16 | 6.923183 | -0.021804 | 16.009 | 15.821 |
| steps 8, max_length 8 | 6.951231 | +0.006244 | 30.713 | 15.821 |

Leitura:

```text
mais steps sem ajuste de LR pode piorar; o proximo sweep deve ajustar scheduler
ou LR por target.
```

### LoRA Rank 1

Comparacao justa em `q_proj`, mesmo texto, `steps=4`, `max_length=8`,
`lr=0.005`:

| metodo | params | initial loss | final loss | loss delta | ganho/param | train CUDA GB |
|---|---:|---:|---:|---:|---:|---:|
| DRM-SAINT-G `q_proj` budget 8192 | 8192 | 6.944987 | 6.923183 | -0.021804 | 2.6616e-06 | 15.821 |
| LoRA rank 1 `q_proj` | 10240 | 6.944987 | 6.944987 | 0.000000 | 0.0000 | 15.813 |

Resultado:

```text
DRM-SAINT-G venceu LoRA rank 1 neste smoke curto em ganho por parametro treinavel.
```

### Veredito

```text
Marco 5 passou.
```

Agora DRM-SAINT-G 14B nao apenas executa treino, mas tambem mostra reducao de loss em
targets especificos, com avaliacao posterior abaixo de 23 GB.

## Proximo Marco

Marco 6 deve tornar a qualidade menos acidental:

- repetir `v_proj` com seeds diferentes;
- testar camadas 0, 1 e 2;
- adicionar scheduler ou reduzir LR quando `steps` aumenta;
- comparar LoRA rank 1 com inicializacao B nao-zero ou update in-place mais fiel;
- testar combinacao de targets `v_proj + o_proj`;
- medir validacao antes/depois no mesmo corpus com mais exemplos.

## Marco 6 - Robustez de Qualidade 14B

Status: **concluido com ressalva de validacao**.

### Mudancas

- adicionado `lr_decay` ao treino DRM-SAINT-G in-place;
- `scripts/benchmark_huggingface_phase15_train_only.py` aceita
  `--lr-decay`;
- LoRA rank 1 agora pode inicializar `B` com valor nao-zero via
  `--lora-b-init-scale`;
- a avaliacao posterior pode medir base e merged no mesmo corpus com
  `--include-base`;
- Marco 6 testou:
  - seeds 31, 32 e 33;
  - camadas 0, 1 e 2;
  - `v_proj`;
  - `v_proj + o_proj`;
  - steps 8 com scheduler.

### Seeds

Config:

```text
target: model.layers.0.self_attn.v_proj.weight
budget: 8192
steps: 4
max_length: 8
learning_rate: 0.005
max_memory: 0=12GiB,cpu=64GiB
```

| seed | DRM-SAINT-G loss delta | DRM-SAINT-G ganho/param | LoRA loss delta | LoRA ganho/param |
|---:|---:|---:|---:|---:|
| 31 | -0.298339 | 3.6418e-05 | +0.012675 | 0.0000 |
| 32 | -0.298339 | 3.6418e-05 | -0.002483 | 4.0419e-07 |
| 33 | -0.298339 | 3.6418e-05 | -0.028358 | 4.6155e-06 |

Leitura:

```text
DRM-SAINT-G foi estavel nos seeds testados; LoRA com B nao-zero melhorou em alguns
seeds, mas ficou abaixo de DRM-SAINT-G no ganho por parametro.
```

### Camadas

Config:

```text
target: model.layers.{0,1,2}.self_attn.v_proj.weight
budget: 8192
steps: 4
max_length: 8
learning_rate: 0.005
seed: 31
```

| camada | DRM-SAINT-G final loss | loss delta | ganho/param | train CUDA GB |
|---:|---:|---:|---:|---:|
| 0 | 6.646648 | -0.298339 | 3.6418e-05 | 15.779 |
| 1 | 6.402409 | -0.542579 | 6.6233e-05 | 15.779 |
| 2 | 6.196043 | -0.748944 | 9.1424e-05 | 15.779 |

Leitura:

```text
o efeito nao ficou restrito a camada 0; neste smoke, camada 2 foi melhor.
```

### Scheduler

Config:

```text
target: model.layers.0.self_attn.v_proj.weight
budget: 8192
steps: 8
learning_rate: 0.005
lr_decay: 0.8
```

Resultado:

| metodo | final loss | loss delta | ganho/param | train_s |
|---|---:|---:|---:|---:|
| DRM-SAINT-G | 6.631443 | -0.313544 | 3.8274e-05 | 30.005 |
| LoRA rank 1 | 6.943999 | -0.000988 | 1.6081e-07 | 30.691 |

Leitura:

```text
decaimento de LR corrigiu a piora vista antes com steps 8 sem scheduler.
```

### Combinacao de Targets

Config:

```text
targets:
  - model.layers.0.self_attn.v_proj.weight
  - model.layers.0.self_attn.o_proj.weight
budget: 16384
steps: 4
learning_rate: 0.005
```

Resultado:

| metodo | final loss | loss delta | ganho/param |
|---|---:|---:|---:|
| DRM-SAINT-G `v_proj + o_proj` | 6.899331 | -0.045656 | 2.7866e-06 |
| LoRA rank 1 | 6.941819 | -0.003168 | 5.1564e-07 |

Leitura:

```text
combinar v_proj + o_proj nao superou treinar apenas v_proj neste teste.
```

### Validacao Com Mais Exemplos

Checkpoint:

```text
runs/phase15_marco6_vproj_layer2/DRM-SAINT-G_b8192_0_12GiB_cpu_64GiB
```

Config:

```text
validation_texts: 4
max_length: 8
max_memory: 0=12GiB,cpu=64GiB
include_base: true
```

Resultado:

| metrica | valor |
|---|---:|
| base_validation_loss | 6.882289 |
| merged_validation_loss | 6.892014 |
| validation_loss_delta | +0.009724 |
| merge load CUDA GB | 20.261 |
| merge eval CUDA GB | 12.500 |

Leitura:

```text
validacao ficou abaixo de 23 GB, mas piorou levemente. O ganho de train loss
ainda nao garante generalizacao.
```

### Veredito

```text
Marco 6 passou para robustez de treino, mas nao fecha qualidade geral.
```

DRM-SAINT-G mostrou ganho consistente em train loss por seed/camada e venceu LoRA
rank 1 nos testes curtos. A validacao com mais exemplos ainda precisa melhorar.

## Proximo Marco

Marco 7 deve focar generalizacao:

- selecionar blocos por mini-validacao, nao apenas por ativacao;
- usar mais de um texto de treino no `train_only`;
- medir validacao durante o treino sem recarregar modelo;
- testar targets em camadas 2 e 3 com validacao como criterio;
- comparar contra LoRA rank 1/2 com o mesmo numero de textos;
- adicionar early stopping por validation loss.

## Marco 7 - Generalizacao 14B

Status: **concluido como diagnostico negativo**.

### Mudancas

- o treino esparso in-place foi separado em
  `saint.adapters.huggingface_sparse_train`;
- `train_only` aceita mais de um texto de treino;
- `train_only` aceita textos de validacao separados;
- a validacao pode ser medida durante o treino, sem recarregar o modelo;
- adicionado early stopping por validation loss;
- LoRA rank 1/2 usa o mesmo numero de textos de treino que DRM-SAINT-G;
- adicionados roteamentos:
  - `validation_gradient`;
  - `validation_magnitude_activation`.

### Roteamento por Mini-Validacao

Configuracao:

```text
model: models/Qwen2.5-14B
target: model.layers.2.self_attn.v_proj.weight
budget: 8192
train_texts: 2
validation_texts: 4
max_length: 8
routing_max_length: 8
max_memory: 0=12GiB,cpu=64GiB
max_cuda_gb: 23
```

Resultado:

| roteamento | status | pico CUDA |
|---|---|---:|
| `validation_gradient`, 12GiB | falhou | 29.962 GB |
| `validation_gradient`, 8GiB | falhou | 29.987 GB |

Leitura:

```text
selecionar blocos por gradiente real de mini-validacao ainda nao cabe no limite
de 23 GB para 14B.
```

### Proxy Barato de Validacao

Para manter a ideia de validacao sem backward completo, foi testado:

```text
validation_magnitude_activation
```

Resultado:

| target | metodo | loss delta | params | pico treino CUDA |
|---|---|---:|---:|---:|
| layer 2 `v_proj` | DRM-SAINT-G validation proxy | +0.073896 | 8192 | 15.782 GB |
| layer 2 `v_proj` | LoRA rank 1 | -0.003053 | 6144 | 15.785 GB |
| layer 2 `v_proj` | LoRA rank 2 | falhou | 12288 | 26.703 GB |
| layer 3 `v_proj` | DRM-SAINT-G validation proxy | +0.058203 | 8192 | 15.782 GB |
| layer 3 `v_proj` | LoRA rank 1 | -0.000484 | 6144 | 15.785 GB |
| layer 3 `v_proj` | LoRA rank 2 | falhou | 12288 | 26.703 GB |

Leitura:

```text
o proxy de validacao cabe em memoria, mas piorou a loss nos targets testados.
LoRA rank 1 melhorou pouco; LoRA rank 2 ainda estoura memoria no caminho atual.
```

### Controle Activation com Multitexto

Para separar o efeito de "mais textos" do efeito do novo roteador, foi repetido
`activation` com os mesmos textos.

Config:

```text
target: model.layers.2.self_attn.v_proj.weight
budget: 8192
train_texts: 2
validation_texts: 4
steps: 4
lr_decay: 0.8
```

Resultado:

| metodo | loss delta | params | ganho/param | pico treino CUDA |
|---|---:|---:|---:|---:|
| DRM-SAINT-G activation | -0.182964 | 8192 | 2.2335e-05 | 15.782 GB |
| LoRA rank 1 | +0.006042 | 6144 | 0.0 | 15.785 GB |
| LoRA rank 2 | falhou | 12288 | 0.0 | 26.703 GB |

Leitura:

```text
o problema nao foi apenas usar mais textos. O roteador activation continuou
melhorando train loss no setup multitexto, enquanto o proxy de validacao piorou.
```

### Veredito

```text
Marco 7 passou como diagnostico, mas nao resolveu generalizacao.
```

O caminho agora tem validacao durante treino e early stopping, mas a escolha de
blocos por mini-validacao ainda precisa de uma tecnica mais barata e mais
fiel. O criterio de fechamento da Fase 15 ainda nao foi atingido, porque o
ganho de treino nao se converteu em validacao melhor de forma consistente.

## Proximo Marco

Marco 8 deve focar em roteamento de validacao barato e competitivo:

- selecionar candidatos por `activation` e ranquear apenas um subconjunto por
  mini-validacao;
- medir ganho real de validacao por bloco sem backward completo quando possivel;
- aplicar rollback de candidatos antes de escolher o delta final;
- tornar LoRA rank 2 esparso/low-rank sem update denso temporario;
- aumentar `train_texts` e `validation_texts` gradualmente;
- manter `activation` como baseline 14B ate o roteador de validacao vencer.

## Marco 8 - Roteamento de Validacao Barato

Status: **em andamento**.

### Mudancas

- adicionado roteamento `activation_validation_rerank`;
- o roteador seleciona candidatos por `activation`;
- apenas um subconjunto dos candidatos e ranqueado por mini-validacao;
- a mini-validacao aplica deltas temporarios por grupos de coordenadas;
- cada grupo usa rollback antes do proximo teste;
- o probe de validacao testa pertubacao nos dois sentidos e usa o melhor ganho;
- LoRA rank 1/2 passou a usar `forward_hook`, sem materializar `A @ B` como
  matriz densa temporaria.

### Nota Sobre LoRA

O baseline LoRA anterior era matematicamente correto, mas operacionalmente
ingenuo para modelos grandes:

```text
update = A @ B
W += update
forward
W -= update
```

Esse caminho cria uma matriz densa temporaria do mesmo tamanho de `W`. Quando
rank 2 estourava memoria, isso nao provava que DRM-SAINT-G era melhor; provava que o
baseline LoRA estava implementado de forma desfavoravel.

O novo baseline usa a forma usual de LoRA no forward:

```text
y = x @ W.T + x @ B.T @ A.T
```

Isto e algebraicamente equivalente a:

```text
W' = W + A @ B
```

mas evita materializar o delta denso completo. Portanto, o nome correto nos
relatorios e:

```text
LoRA forward-hook baseline
```

ou:

```text
LoRA sem materializacao densa do delta
```

Esse baseline e mais justo. Ele aumenta a barra para DRM-SAINT-G, porque compara
contra uma implementacao mais proxima do uso real de LoRA/PEFT.

### Resultados Parciais

Config principal:

```text
model: models/Qwen2.5-14B
target: model.layers.2.self_attn.v_proj.weight
budget: 8192
train_texts: 3
validation_texts: 6
steps: 4
max_length: 8
max_memory: 0=12GiB,cpu=64GiB
```

Comparacao inicial:

| metodo | loss delta | params | ganho/param | pico treino CUDA |
|---|---:|---:|---:|---:|
| DRM-SAINT-G `activation_validation_rerank` | -0.090874 | 8192 | 1.1093e-05 | 15.782 GB |
| DRM-SAINT-G `activation` | -0.086226 | 8192 | 1.0526e-05 | 15.782 GB |
| LoRA rank 1 forward-hook | -0.132801 | 6144 | 2.1615e-05 | 15.778 GB |
| LoRA rank 2 forward-hook | -0.573527 | 12288 | 4.6674e-05 | 15.778 GB |

Leitura:

```text
o rerank por mini-validacao melhorou ligeiramente contra activation nesse
teste, mas ainda perdeu para LoRA rank 1/2.
```

O resultado e util porque corrige dois pontos:

- DRM-SAINT-G agora tem um roteador de validacao barato que cabe no limite de memoria;
- LoRA rank 2 deixou de ser desclassificado por detalhe de implementacao.

### Veredito Parcial

```text
Marco 8 ainda nao fecha.
```

O progresso tecnico e real, mas o resultado cientifico ficou mais exigente:
DRM-SAINT-G precisa vencer um LoRA forward-hook forte, ou mostrar vantagem clara em
checkpoint, memoria, ganho por parametro treinavel ou comportamento em budgets
menores.

## Proximo Marco

O proximo passo deve atacar onde DRM-SAINT-G ainda perde para LoRA:

- trocar o delta por coordenada independente por blocos 2x2/4x4 treinaveis;
- usar o rerank de validacao para escolher blocos, nao valores isolados;
- comparar budgets menores, por exemplo 1024, 2048 e 4096;
- medir validacao real antes/depois no mesmo corpus, nao apenas train loss;
- testar `activation_validation_rerank` em camadas 1, 2 e 3;
- manter LoRA forward-hook rank 1/2 como baseline obrigatorio.

## Marco 9 - Blocos Treinaveis por Validacao

Status: **concluido como experimento inicial**.

### Mudancas

- adicionado roteamento `activation_block_validation_rerank`;
- adicionado `--routing-block-size`;
- blocos 2x2 e 4x4 podem ser selecionados por score de ativacao;
- a mini-validacao ranqueia blocos, nao apenas coordenadas individuais;
- cada bloco testado aplica delta temporario e faz rollback;
- adicionado `--validation-rerank-max-candidates` para limitar custo;
- checkpoints e resultados agora registram:
  - `initial_validation_loss`;
  - `validation_loss`;
  - `validation_loss_delta`.

### Smoke 3B

Config:

```text
model: models/Qwen2.5-3B
target: layer0 v_proj
routing: activation_block_validation_rerank
block_size: 2
budget: 64
train_texts: 2
validation_texts: 2
```

Resultado:

| metodo | train delta | validation delta | params | routing CUDA |
|---|---:|---:|---:|---:|
| DRM-SAINT-G block2 | +0.012564 | +0.003355 | 32 | 6.194 GB |
| LoRA rank 1 forward-hook | -0.197236 | n/a | 2304 | 6.262 GB |

Leitura:

```text
o caminho por bloco executa e salva metricas de validacao reais, mas o smoke 3B
nao mostrou ganho.
```

### 14B - Budget Pequeno por Bloco

Config principal:

```text
model: models/Qwen2.5-14B
target: model.layers.2.self_attn.v_proj.weight
routing: activation_block_validation_rerank
block_size: 4
budget: 1024
validation_rerank_max_candidates: 8
train_texts: 3
validation_texts: 6
steps: 4
max_memory: 0=12GiB,cpu=64GiB
```

Resultado com metrica de validacao corrigida:

| metodo | train delta | validation delta | params | routing_s | train CUDA |
|---|---:|---:|---:|---:|---:|
| DRM-SAINT-G block4 layer 2 | -0.012179 | -0.013615 | 128 | 34.756 | 15.781 GB |
| LoRA rank 1 forward-hook | -0.227059 | n/a | 6144 | n/a | 15.778 GB |

Leitura:

```text
DRM-SAINT-G block4 melhorou validacao com apenas 128 parametros efetivos, mas ainda
perde muito em qualidade absoluta para LoRA rank 1.
```

### Camadas 1, 2 e 3

Config:

```text
block_size: 4
validation_rerank_max_candidates: 8
budget: 1024
```

| camada | train delta DRM-SAINT-G | validation delta DRM-SAINT-G | params efetivos | routing_s | LoRA rank 1 train delta |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.019991 | n/a | 128 | 35.212 | -0.159320 |
| 2 | -0.012179 | -0.013615 | 128 | 34.756 | -0.227059 |
| 3 | -0.022025 | n/a | 128 | 36.826 | -0.196929 |

Observacao:

```text
as primeiras medicoes de camadas 1 e 3 foram feitas antes da correcao do campo
validation_loss final; por isso a tabela usa somente train delta nelas.
```

### Budget e Custo

O teste com `block_size=2`, `budget=1024` e `max_candidates=64` selecionou 64
blocos, ou 256 parametros efetivos:

| metodo | train delta | params efetivos | routing_s |
|---|---:|---:|---:|
| DRM-SAINT-G block2 layer 2 | +0.026165 | 256 | 242.675 |

O teste com `block_size=4`, `budget=1024` e 64 candidatos usou o budget completo,
mas ainda ficou caro:

| metodo | train delta | params efetivos | routing_s |
|---|---:|---:|---:|
| DRM-SAINT-G block4 layer 2 | -0.014973 | 1024 | 238.497 |

Leitura:

```text
o rerank por validacao e viavel em memoria, mas caro em tempo. O limite de
candidatos e necessario para experimentos 14B.
```

### Veredito

```text
Marco 9 introduziu blocos treinaveis e validacao real, mas ainda nao vence LoRA.
```

O resultado mais promissor e:

```text
DRM-SAINT-G block4 layer 2:
validation_loss_delta = -0.013615
params efetivos = 128
train CUDA = 15.781 GB
```

Isso mostra sinal de generalizacao com poucos parametros, mas o ganho ainda e
pequeno. LoRA forward-hook segue baseline obrigatorio e mais forte.

## Proximo Marco

Marco 10 deve melhorar eficiencia e qualidade dos blocos:

- treinar blocos com valor estruturado, nao valores independentes por
  coordenada;
- testar bloco com escala + prototipo, por exemplo `delta = scale * P_4x4`;
- acumular varios blocos candidatos em um unico forward de validacao para
  reduzir `routing_s`;
- selecionar mais blocos sem aumentar linearmente o numero de forwards;
- medir validacao final para camadas 1, 2 e 3 com a metrica corrigida;
- comparar contra LoRA forward-hook em validation loss, nao apenas train loss.

## Marco 10 - Blocos Estruturados

Status: **concluido como diagnostico**.

### Mudancas

- adicionado `activation_structured_block_validation_rerank`;
- blocos estruturados usam um parametro treinavel por bloco:

```text
delta_bloco = scale * prototype
```

- o prototipo inicial usa o sinal do peso base no bloco;
- o treino atualiza apenas `scale`;
- adicionado `--validation-rerank-batch-size`;
- varios blocos candidatos podem ser testados no mesmo forward de validacao;
- LoRA forward-hook agora registra:
  - `initial_validation_loss`;
  - `validation_loss`;
  - `validation_loss_delta`.

### Smoke 3B

Config:

```text
model: models/Qwen2.5-3B
routing: activation_structured_block_validation_rerank
block_size: 4
budget: 16
validation_rerank_max_candidates: 8
validation_rerank_batch_size: 4
```

Resultado:

| metodo | train delta | validation delta | params | routing_s |
|---|---:|---:|---:|---:|
| DRM-SAINT-G structured block4 | +0.065132 | -0.032646 | 8 | 0.819 |
| LoRA rank 1 forward-hook | -0.289929 | n/a | 2304 | n/a |

Leitura:

```text
o bloco estruturado consegue melhorar validacao com poucos parametros, mas pode
piorar train loss. Isso confirma que a metrica de validacao virou necessaria.
```

### 14B - Camadas 1, 2 e 3

Config:

```text
model: models/Qwen2.5-14B
target: v_proj
routing: activation_structured_block_validation_rerank
block_size: 4
budget: 16
validation_rerank_max_candidates: 16
validation_rerank_batch_size: 8
train_texts: 3
validation_texts: 6
steps: 4
```

Resultados DRM-SAINT-G:

| camada | train delta | validation delta | params | routing_s | train CUDA |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.003109 | +0.002546 | 16 | 16.888 | 15.781 GB |
| 2 | -0.004762 | -0.002882 | 16 | 16.282 | 15.781 GB |
| 3 | -0.025127 | +0.021051 | 16 | 16.304 | 15.781 GB |

Comparacao LoRA forward-hook:

| camada | LoRA rank 1 val delta | LoRA rank 2 val delta | LoRA rank 1 params | LoRA rank 2 params |
|---:|---:|---:|---:|---:|
| 1 | -0.116323 | -0.180719 | 6144 | 12288 |
| 2 | -0.285410 | -0.197755 | 6144 | 12288 |
| 3 | -0.255529 | -0.450788 | 6144 | 12288 |

### Eficiencia de Roteamento

O Marco 9, com blocos livres e 8 candidatos, tinha custo aproximado:

```text
routing_s ~= 35s
```

O Marco 10, com blocos estruturados e `validation_rerank_batch_size=8`,
reduziu para:

```text
routing_s ~= 16s
```

Leitura:

```text
agrupar candidatos em um mesmo forward reduziu o custo de roteamento pela
metade, sem aumentar o pico de VRAM.
```

### Veredito

```text
Marco 10 melhorou eficiencia, mas nao resolveu qualidade.
```

O resultado mais util:

```text
layer 2:
DRM-SAINT-G structured block4 validation_delta = -0.002882
params = 16
routing_s = 16.282
```

Isso e um sinal pequeno de generalizacao com pouquissimos parametros, mas ainda
esta muito abaixo de LoRA forward-hook em validation loss.

## Proximo Marco

Marco 11 deve aumentar capacidade sem perder compressao:

- usar mais de um prototipo por bloco, por exemplo
  `delta = scale_a * P_a + scale_b * P_b`;
- inicializar prototipos por gradiente/ativacao, nao pelo sinal do peso base;
- permitir `scale` por linha ou coluna dentro do bloco;
- testar budgets estruturados 32, 64 e 128 scales;
- manter rerank batelado;
- comparar ganho por parametro contra LoRA rank 1 em validation loss;
- decidir se DRM-SAINT-G 14B deve focar em compressao extrema ou qualidade contra
  LoRA.

## Marco 11 - Multiplos Prototipos Estruturados

Status: **concluido como sinal de compressao extrema**.

### Mudancas

- blocos estruturados agora aceitam multiplos prototipos por bloco:

```text
delta = scale_a * P_a + scale_b * P_b
```

- adicionado `--structured-prototype-count`;
- adicionado `--structured-prototype-mode`;
- adicionado `--structured-scale-granularity`;
- modos iniciais:
  - `weight_sign`;
  - `activation`;
  - `weight_activation`;
  - `weight_value`;
- escalas podem ser:
  - por bloco;
  - por linha;
  - por coluna;
- metadata de delta registra estrutura efetiva:
  - `prototype_count`;
  - `prototype_shape`;
  - `scale_id_shape`;
  - `scale_count`.

### Config Principal

```text
model: models/Qwen2.5-14B
target: model.layers.2.self_attn.v_proj.weight
routing: activation_structured_block_validation_rerank
block_size: 4
prototype_count: 2
prototype_mode: weight_activation
scale_granularity: row
validation_rerank_batch_size: 8
validation_rerank_max_candidates: 32
train_texts: 3
validation_texts: 6
steps: 4
```

### Budgets Estruturados

| budget scales | validation delta | train delta | routing_s | train CUDA |
|---:|---:|---:|---:|---:|
| 32 | -0.021731 | -0.005986 | 28.497 | 15.781 GB |
| 64 | -0.007499 | -0.028401 | 48.258 | 15.782 GB |
| 128 | -0.003388 | -0.003791 | 92.410 | 15.782 GB |

Observacao:

```text
budget 32 foi o melhor ponto de validacao. Aumentar budget nao melhorou neste
setup, provavelmente porque o rerank/prototipo ainda seleciona blocos fracos
ou redundantes.
```

### Comparacao Contra LoRA

Baseline no mesmo alvo/corpus:

| metodo | validation delta | params | ganho val/param |
|---|---:|---:|---:|
| DRM-SAINT-G structured budget 32 | -0.021731 | 32 | 6.79e-04 |
| LoRA rank 1 forward-hook | -0.165617 | 6144 | 2.70e-05 |

Leitura:

```text
LoRA rank 1 vence em qualidade absoluta.
DRM-SAINT-G budget 32 vence em ganho de validacao por parametro.
```

Isso e o primeiro resultado 14B mais claro para DRM-SAINT-G nesta fase:

```text
DRM-SAINT-G nao esta competindo com LoRA em qualidade total,
mas mostra eficiencia extrema por parametro em budget muito pequeno.
```

### Veredito

```text
Marco 11 sugere que DRM-SAINT-G 14B deve focar compressao extrema primeiro.
```

A direcao mais promissora nao e tentar bater LoRA rank 1 em loss absoluta com
dezenas de parametros. O eixo onde DRM-SAINT-G tem sinal e:

```text
ganho por parametro,
checkpoint pequeno,
delta local,
memoria controlada.
```

## Proximo Marco

Marco 12 deve consolidar DRM-SAINT-G como metodo de compressao extrema:

- reportar sempre `validation_gain_per_parameter`;
- comparar DRM-SAINT-G budget 16/32/64 contra LoRA rank 1 truncado por parametro
  equivalente quando possivel;
- testar mais camadas com budget 32;
- testar `prototype_mode=activation` e `weight_value`;
- testar `scale_granularity=block`, `row` e `col`;
- selecionar automaticamente o melhor ponto por ganho de validacao por
  parametro;
- decidir criterio de passagem da Fase 15 como eficiencia por parametro, nao
  qualidade absoluta contra LoRA.

## Marco 12 - Eficiencia por Parametro

Status: **concluido como criterio de compressao extrema**.

### Mudancas

- criado `scripts/benchmark_huggingface_phase15_efficiency.py`;
- o sweep registra sempre `validation_gain_per_parameter`;
- o melhor ponto DRM-SAINT-G e selecionado automaticamente por ganho de validacao por
  parametro;
- adicionado comparativo contra LoRA rank 1 por estimativa de parametro
  equivalente;
- corrigido o caminho `train_only + measure_loss` para sempre calcular
  `validation_loss` final real, mesmo sem `--validate-during-train`;
- o criterio da Fase 15 passa a ser eficiencia por parametro/checkpoint, nao
  loss absoluta contra LoRA.

### Observacao Sobre a Metrica

Uma rodada inicial retornou `validation_loss = 0.0` quando a validacao durante
treino estava desligada. Isso era um bug de medicao, nao um ganho real.

Correcao:

```text
train_only + measure_loss agora calcula a validation_loss final aplicando o
delta in-place no batch de validacao.
```

### Budget 16/32/64

Config:

```text
model: models/Qwen2.5-14B
target: model.layers.2.self_attn.v_proj.weight
routing: activation_structured_block_validation_rerank
prototype_count: 2
prototype_mode: activation
scale_granularity: row
block_size: 4
steps: 4
max_length: 4
train_texts: 3
validation_texts: 6
gradient_checkpointing: on
```

Resultados:

| budget | validation delta | ganho val/param | params | train CUDA | routing_s |
|---:|---:|---:|---:|---:|---:|
| 16 | +0.011784 | 0.000000e+00 | 16 | 17.412 GB | 23.299 |
| 32 | -0.006461 | 2.019107e-04 | 32 | 17.412 GB | 24.368 |
| 64 | +0.013001 | 0.000000e+00 | 64 | 17.412 GB | 42.422 |

LoRA rank 1 no mesmo alvo:

| metodo | validation delta | ganho val/param | params |
|---|---:|---:|---:|
| LoRA rank 1 forward-hook | -0.004622 | 7.521982e-07 | 6144 |
| DRM-SAINT-G budget 32 | -0.006461 | 2.019107e-04 | 32 |

Leitura:

```text
DRM-SAINT-G budget 32 venceu LoRA rank 1 em validation delta nesta rodada curta e
venceu com folga em ganho de validacao por parametro.
```

O ponto importante nao e afirmar superioridade geral contra LoRA. A amostra e
pequena. O resultado suporta um criterio mais especifico:

```text
DRM-SAINT-G pode ser muito forte em budgets extremos, onde LoRA rank 1 ainda tem
milhares de parametros.
```

### Modos e Granularidades

Sweep em layer 2, budget 32:

| prototype_mode | scale_granularity | validation delta | ganho val/param |
|---|---|---:|---:|
| activation | block | -0.006461 | 2.019107e-04 |
| activation | row | -0.006461 | 2.019107e-04 |
| activation | col | -0.006461 | 2.019107e-04 |
| weight_value | block | -0.006461 | 2.019107e-04 |
| weight_value | row | -0.006461 | 2.019107e-04 |
| weight_value | col | -0.006461 | 2.019107e-04 |

Leitura:

```text
nesta configuracao curta, prototype_mode e scale_granularity nao mudaram o
resultado observado. Isso sugere que a selecao dos blocos dominou mais do que a
forma interna do prototipo.
```

### Mais Camadas

Budget 32, `prototype_mode=activation`, `scale_granularity=row`:

| camada | validation delta | ganho val/param | params |
|---:|---:|---:|---:|
| 1 | -0.019947 | 6.233305e-04 | 32 |
| 2 | -0.006461 | 2.019107e-04 | 32 |
| 3 | +0.002833 | 0.000000e+00 | 32 |

Melhor ponto:

```text
layer 1, v_proj, budget 32
validation_delta = -0.019947
validation_gain_per_parameter = 6.233305e-04
```

### Veredito da Fase 15

```text
Fase 15 passou no criterio de eficiencia extrema por parametro.
```

Ela ainda nao passou como substituto geral de LoRA em qualidade absoluta. O
criterio correto para prosseguir e:

```text
DRM-SAINT-G deve ser avaliado primeiro como adaptador ultra-compacto,
com checkpoint minimo e ganho alto por parametro.
```

### Direcao Conceitual

A leitura mais promissora para o proximo ciclo e aproximar DRM-SAINT-G de:

```text
Delta W = A Phi B
```

Onde `A` e `B` sao projecoes pequenas, mas `Phi` nao e apenas uma matriz densa
de baixa dimensao. Em DRM-SAINT-G, `Phi` deve representar:

- operador geometrico local;
- manifold de microblocos;
- bloco relacional 2x2 ou 4x4;
- codebook topologico;
- composicao multi-escala de operadores.

Isso separa DRM-SAINT-G de LoRA classico:

```text
LoRA procura baixa patente linear.
DRM-SAINT-G deve procurar geometria relacional local reutilizavel.
```

## Proximo Marco

Marco 13 deve transformar o bloco estruturado em operador relacional:

- implementar variante `DRM-SAINT-G_phi_delta`;
- testar `Delta W = A Phi B` em um unico alvo 14B;
- manter `Phi` pequeno, por exemplo 2x2/4x4 ou composicao de microblocos;
- comparar contra o melhor ponto do Marco 12:
  `layer 1 v_proj budget 32`;
- medir validation_gain_per_parameter, checkpoint size e train CUDA;
- decidir se a Fase 16 deve escalar o formato atual ou o formato `Phi`.

## Marco 13 - DRM-SAINT-G Phi Delta

Status: **concluido como novo melhor ponto de eficiencia**.

### Ideia

Implementar uma variante DRM-SAINT-G baseada em:

```text
Delta W = A Phi B
```

Nesta primeira implementacao, para manter memoria controlada em 14B, `A` e `B`
sao obtidos por SVD local do bloco escolhido:

```text
A = U_r
B = V_r^T
Phi = operador pequeno treinavel
```

O delta nao e materializado como matriz densa. Cada base de `Phi` produz um
prototipo local:

```text
prototipo_k = A basis_k B
delta_bloco = soma theta_k * prototipo_k
```

Assim o treino continua usando o caminho in-place esparso ja existente.

### Entregas

- criado `saint/adapters/huggingface_phi_routing.py`;
- criado `scripts/benchmark_huggingface_phase15_phi.py`;
- adicionado routing method `activation_phi_validation_rerank`;
- adicionado `--phi-rank`;
- adicionado `--phi-variant`;
- variantes iniciais de `Phi`:
  - `dense`;
  - `diagonal`;
  - `upper_triangular`;
  - `block_diagonal`;
  - `kronecker`;
  - `hadamard`;
  - `codebook_2x2`;
  - `codebook_4x4`.

### Smoke

O smoke em `models/sshleifer_tiny_gpt2` executou todas as variantes de `Phi`
de ponta a ponta.

### Comparacao Contra Marco 12

Baseline Marco 12:

```text
target: model.layers.1.self_attn.v_proj.weight
budget: 32
validation_gain_per_parameter: 6.233305e-04
```

Config Marco 13:

```text
model: models/Qwen2.5-14B
target: model.layers.1.self_attn.v_proj.weight
routing: activation_phi_validation_rerank
phi_rank: 4
budget: 32
block_size: 4
steps: 4
train_texts: 3
validation_texts: 6
max_length: 4
gradient_checkpointing: on
```

Resultados:

| Phi | validation delta | ganho val/param | params | checkpoint | routing_s |
|---|---:|---:|---:|---:|---:|
| dense | +0.000049 | 0.000000e+00 | 32 | 2868 B | 9.520 |
| diagonal | +0.003608 | 0.000000e+00 | 32 | 10425 B | 23.403 |
| upper_triangular | +0.013865 | 0.000000e+00 | 30 | 4155 B | 14.315 |
| block_diagonal | +0.024908 | 0.000000e+00 | 32 | 5392 B | 13.987 |
| kronecker | +0.000049 | 0.000000e+00 | 32 | 2867 B | 9.219 |
| hadamard | -0.023148 | 7.715861e-04 | 30 | 12820 B | 23.299 |
| codebook_2x2 | +0.004420 | 0.000000e+00 | 32 | 10398 B | 22.817 |
| codebook_4x4 | -0.012781 | 3.993958e-04 | 32 | 40585 B | 23.466 |

Melhor ponto:

```text
Phi = hadamard
validation_delta = -0.023148
validation_gain_per_parameter = 7.715861e-04
params = 30
```

Comparacao:

| metodo | validation delta | ganho val/param | params |
|---|---:|---:|---:|
| Marco 12 structured block | -0.019947 | 6.233305e-04 | 32 |
| Marco 13 Phi hadamard | -0.023148 | 7.715861e-04 | 30 |

### Veredito

```text
DRM-SAINT-G Phi superou o melhor ponto do Marco 12 em ganho por parametro.
```

Isso reforca a leitura dos PDFs:

```text
DRM-SAINT-G fica mais promissor quando Phi e tratado como operador estrutural,
nao como LoRA pequeno nem como bloco livre.
```

O resultado mais interessante foi `hadamard`, porque ele nao e o mais flexivel.
Ele tem poucos graus de liberdade e mesmo assim foi o melhor, sugerindo que a
estrutura do operador importa mais que a capacidade bruta neste budget.

## Proximo Marco

Marco 14 deve consolidar `Phi`:

- repetir `Phi=hadamard` com seeds 31, 32 e 33;
- testar `layer 1`, `layer 2` e `layer 3`;
- testar `phi_rank` 2, 4 e 8;
- comparar contra LoRA rank 1 no mesmo corpus;
- testar `Phi` inicializado por gradiente, nao so por SVD local do peso;
- medir se `Phi` mantem vantagem quando `train_texts` e `validation_texts`
  aumentam.

## Marco 14 - Confirmacao do Phi Hadamard

Status: **concluido como confirmacao parcial**.

### Objetivo

Verificar se o melhor resultado do Marco 13 nao foi apenas um ponto acidental.

O foco foi:

- `Phi = hadamard`;
- `budget = 32`;
- `target = v_proj`;
- comparacao contra o melhor Marco 12;
- seeds, layers e ranks diferentes;
- LoRA rank 1 como controle.

### Seeds

Config:

```text
layer: 1
phi_rank: 4
seeds: 31, 32, 33
```

Resultados:

| seed | validation delta | ganho val/param | params |
|---:|---:|---:|---:|
| 31 | -0.009962 | 3.320694e-04 | 30 |
| 32 | -0.016432 | 5.477269e-04 | 30 |
| 33 | -0.008009 | 2.669652e-04 | 30 |

Leitura:

```text
Phi hadamard rank 4 melhorou validacao em todas as seeds, mas nao reproduziu
o pico do Marco 13 nem bateu o Marco 12 em ganho por parametro.
```

### Layers

Config:

```text
seed: 31
phi_rank: 4
layers: 1, 2, 3
```

Resultados:

| layer | validation delta | ganho val/param | params |
|---:|---:|---:|---:|
| 1 | -0.016432 | 5.477269e-04 | 30 |
| 2 | -0.014893 | 4.964193e-04 | 30 |
| 3 | -0.016785 | 5.594889e-04 | 30 |

Leitura:

```text
o sinal positivo aparece em mais de uma camada, mas rank 4 ainda fica abaixo
do criterio Marco 12.
```

### Ranks

Config:

```text
seed: 31
layer: 1
phi_ranks: 2, 4, 8
```

Resultados:

| phi_rank | validation delta | ganho val/param | params |
|---:|---:|---:|---:|
| 2 | +0.022626 | 0.000000e+00 | 30 |
| 4 | -0.008009 | 2.669652e-04 | 30 |
| 8 | -0.023454 | 7.817904e-04 | 30 |

Leitura:

```text
rank 8 foi o primeiro ponto do Marco 14 que bateu o Marco 12.
```

### Phi Source

Teste:

```text
phi_source: score
seed: 31
layer: 1
phi_rank: 8
```

Resultado:

| phi_source | validation delta | ganho val/param |
|---|---:|---:|
| score | +0.004720 | 0.000000e+00 |

Leitura:

```text
usar o score de ativacao como fonte direta para SVD local piorou neste teste.
O melhor caminho continua sendo SVD local do peso.
```

### LoRA Rank 1

Controle:

```text
seed: 31
layer: 1
phi_rank: 8
LoRA rank 1
```

Resultados:

| metodo | validation delta | ganho val/param | params |
|---|---:|---:|---:|
| DRM-SAINT-G Phi hadamard rank 8 | -0.032917 | 1.097218e-03 | 30 |
| LoRA rank 1 forward-hook | +0.004285 | 0.000000e+00 | 6144 |

Leitura:

```text
nesta rodada curta, DRM-SAINT-G Phi rank 8 venceu LoRA rank 1 em validation loss e
em ganho por parametro.
```

Essa comparacao ainda e pequena. Ela nao prova superioridade geral contra LoRA,
mas fortalece o eixo cientifico correto:

```text
DRM-SAINT-G Phi e competitivo em budgets extremos.
```

### Veredito

```text
Marco 14 confirma Phi hadamard como direcao valida, mas muda o candidato
principal de rank 4 para rank 8.
```

Resumo:

- rank 4 e estavel, mas nao bate Marco 12;
- rank 8 bate Marco 12 e LoRA rank 1 em uma rodada curta;
- `phi_source=score` nao ajudou;
- SVD local do peso continua melhor nesta etapa;
- o ganho aparece em layers 1, 2 e 3.

## Proximo Marco

Marco 15 deve testar robustez real do `Phi rank 8`:

- seeds 31, 32 e 33 com `phi_rank=8`;
- LoRA rank 1 com seeds 31, 32 e 33;
- aumentar `train_texts` e `validation_texts`;
- testar `steps` 8 e scheduler;
- testar `gradient_phi_validation_rerank`;
- comparar `v_proj`, `o_proj` e `v_proj + o_proj`;
- decidir se Fase 15 fecha com `Phi hadamard rank 8` como baseline para Fase
  16.

## Marco 15 - Robustez do Phi Rank 8

Status: **concluido com ressalvas**.

### Objetivo

Testar se `Phi hadamard rank 8` continua sendo o melhor candidato quando:

- seeds variam;
- LoRA rank 1 usa as mesmas seeds;
- o corpus curto aumenta;
- o numero de steps aumenta;
- o roteamento usa gradiente;
- os alvos mudam para `v_proj`, `o_proj` e `v_proj + o_proj`.

### Seeds 31, 32 e 33

Config:

```text
model: models/Qwen2.5-14B
target: layer 1 v_proj
phi_rank: 8
budget: 32
steps: 4
train_texts: 4
validation_texts: 8
max_length: 4
```

Resultados DRM-SAINT-G Phi:

| seed | validation delta | ganho val/param | params |
|---:|---:|---:|---:|
| 31 | -0.012927 | 4.309018e-04 | 30 |
| 32 | -0.012927 | 4.309018e-04 | 30 |
| 33 | -0.012927 | 4.309018e-04 | 30 |

Resultados LoRA rank 1:

| seed | validation delta | ganho val/param | params |
|---:|---:|---:|---:|
| 31 | -0.014132 | 2.300212e-06 | 6144 |
| 32 | -0.000898 | 1.462176e-07 | 6144 |
| 33 | -0.003596 | 5.852586e-07 | 6144 |

Leitura:

```text
LoRA rank 1 teve validation delta absoluto parecido ou melhor em uma seed, mas
DRM-SAINT-G Phi venceu com folga em ganho por parametro.
```

Media:

| metodo | mean validation delta | mean ganho val/param |
|---|---:|---:|
| DRM-SAINT-G Phi rank 8 | -0.012927 | 4.309018e-04 |
| LoRA rank 1 | -0.006209 | 1.010563e-06 |

### Steps 8 e Scheduler

Config:

```text
steps: 8
lr_decay: 0.9
train_texts: 4
validation_texts: 8
```

Resultado:

| metodo | validation delta | ganho val/param |
|---|---:|---:|
| DRM-SAINT-G Phi rank 8 | +0.008677 | 0.000000e+00 |

Leitura:

```text
mais steps com scheduler piorou validacao neste regime. O baseline atual deve
continuar com steps curtos.
```

### Gradient Phi Routing

Config:

```text
routing: gradient_phi_validation_rerank
validation_rerank_max_candidates: 16
```

Resultado:

```text
falhou por VRAM:
CUDA budget exceeded during routing: 29.672 GB
```

Leitura:

```text
roteamento por gradiente ainda nao e viavel no limite de 23 GB. O baseline da
Fase 16 deve continuar usando activation_phi_validation_rerank.
```

### Alvos

Config:

```text
seed: 31
layer: 1
phi_rank: 8
targets: o_proj, v_proj + o_proj
```

Resultados:

| target | validation delta | ganho val/param | params |
|---|---:|---:|---:|
| o_proj | +0.000000 | 0.000000e+00 | 30 |
| v_proj + o_proj | -0.012927 | 4.309018e-04 | 30 |

Leitura:

```text
o_proj isolado nao ajudou. v_proj + o_proj reproduziu o ganho de v_proj, mas
nao melhorou o resultado, indicando que o roteador ainda escolhe essencialmente
o mesmo eixo util.
```

### Veredito

```text
Fase 15 fecha com Phi hadamard rank 8 como baseline de 14B para Fase 16,
mas nao como prova definitiva de qualidade contra LoRA.
```

O que foi demonstrado:

- roda em Qwen2.5-14B abaixo do limite de VRAM definido;
- checkpoint do delta fica pequeno, cerca de 12-13 KB nos runs principais;
- ganho por parametro e muito superior ao LoRA rank 1 neste corpus curto;
- `v_proj` e o alvo util;
- `steps=4` e melhor que `steps=8` neste regime;
- roteamento por ativacao e o caminho viavel;
- roteamento por gradiente precisa de reducao de memoria antes de virar
  baseline.

O que nao foi demonstrado:

- superioridade geral de quality/loss absoluta contra LoRA;
- robustez em corpus maior;
- vantagem com mais steps;
- beneficio real de `o_proj` isolado.

Baseline historico de 14B para a antiga Fase 16:

```text
routing: activation_phi_validation_rerank
phi_variant: hadamard
phi_rank: 8
phi_source: weight
target: v_proj
budget: 32
steps: 4
train_texts: 4
validation_texts: 8
```

## Proxima Fase

Com a reorganizacao, a nova Fase 16 deve primeiro criar uma comparacao full
controlada que caiba na RTX 4090. A escala 70B passa para a Fase 17.

A Fase 17 deve iniciar a escala 70B com o baseline acima, mas com foco em
infraestrutura:

- carregar 70B com offload/quantizacao;
- rodar smoke sem treino;
- aplicar `Phi hadamard rank 8` em `v_proj`;
- manter budget pequeno;
- medir load, routing, train, checkpoint e merge;
- comparar contra LoRA/QLoRA apenas se couber no mesmo limite.
