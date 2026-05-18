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
