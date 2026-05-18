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

## Proximo Marco

Marco 3 deve melhorar a competitividade SAINT em GPT-2 small:

- selecionar deltas por gradiente real, nao por magnitude inicial;
- testar mais matrizes alvo por camada;
- aumentar steps e validar se SAINT ganha com treino mais longo;
- comparar budgets maiores sem aumentar payload denso;
- reduzir overhead CUDA do forward funcional;
- manter LoRA rank `2` e `4` como controle minimo.
