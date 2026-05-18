# Fase 15 - Escala 14B

Status: **em andamento**.

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

### Tentativa de Treino SAINT

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
- treino SAINT 14B ainda nao e viavel neste caminho;
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
- so tentar LoRA 14B depois que SAINT fizer um step completo abaixo de 5 minutos.

## Marco 2 - Alvo Unico em Camada Residente

Status: **concluido como diagnostico, ainda sem treino viavel**.

### Mudancas

- `saint.adapters.huggingface_forward` aceita `target_names` explicito;
- o runtime SAINT aceita filtrar matriz alvo por `target_device`;
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

### Tentativa SAINT Limitada

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
- LoRA 14B continua adiado, porque SAINT ainda nao completou um step abaixo de
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
- so reativar LoRA 14B depois de um step SAINT completo abaixo de 5 minutos.

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
deltas.saintdelta.json
optimizer.saintopt
checkpoint.json
metrics.json
logs.jsonl
```

### Veredito

```text
Marco 3 passou.
```

SAINT completou um step autograd 14B em modo train-only abaixo de 5 minutos,
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
- cada ponto SAINT roda em subprocesso separado para evitar residuos de CUDA;
- o comparador gera:
  - `phase15_compare_results.json`;
  - `phase15_compare_results.md`;
- criado `scripts/benchmark_huggingface_phase15_eval_checkpoint.py`;
- a avaliacao posterior roda em processo separado a partir do checkpoint SAINT;
- LoRA rank 1 foi testado somente depois de SAINT passar no limite de CUDA.

### Resultado SAINT

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
SAINT budget 8192 @ 14GiB: train_s 3.486, train CUDA 17.401 GB
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
runs/phase15_marco4_qwen25_14b_compare_isolated/saint_b8192_0_16GiB_cpu_64GiB
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

SAINT agora tem uma curva inicial de budgets em 14B e um primeiro baseline LoRA
rank 1 no mesmo alvo. A avaliacao posterior existe, mas precisa reduzir pico
antes de virar criterio de fechamento.

## Proximo Marco

Marco 5 deve melhorar qualidade e avaliacao:

- reduzir pico da avaliacao posterior abaixo de 23 GB;
- medir loss antes/depois no mesmo processo train-only sem duplicar memoria;
- repetir com mais steps, por exemplo 4 e 8;
- testar janela maior, por exemplo `max_length` 8 e 16;
- comparar SAINT budget 8192/16384 contra LoRA rank 1 em loss delta;
- salvar ganho por parametro treinavel real;
- testar target em `v_proj` e `o_proj`, nao apenas `q_proj`.

## Marco 5 - Qualidade e Avaliacao 14B

Status: **concluido**.

### Mudancas

- o modo `train_only` agora pode medir loss antes/depois no mesmo processo com
  `--measure-loss`;
- `initial_loss`, `loss_delta` e `gain_per_parameter` sao registrados para
  SAINT e LoRA;
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
| SAINT `q_proj` budget 8192 | 8192 | 6.944987 | 6.923183 | -0.021804 | 2.6616e-06 | 15.821 |
| LoRA rank 1 `q_proj` | 10240 | 6.944987 | 6.944987 | 0.000000 | 0.0000 | 15.813 |

Resultado:

```text
SAINT venceu LoRA rank 1 neste smoke curto em ganho por parametro treinavel.
```

### Veredito

```text
Marco 5 passou.
```

Agora SAINT 14B nao apenas executa treino, mas tambem mostra reducao de loss em
targets especificos, com avaliacao posterior abaixo de 23 GB.

## Proximo Marco

Marco 6 deve tornar a qualidade menos acidental:

- repetir `v_proj` com seeds diferentes;
- testar camadas 0, 1 e 2;
- adicionar scheduler ou reduzir LR quando `steps` aumenta;
- comparar LoRA rank 1 com inicializacao B nao-zero ou update in-place mais fiel;
- testar combinacao de targets `v_proj + o_proj`;
- medir validacao antes/depois no mesmo corpus com mais exemplos.
