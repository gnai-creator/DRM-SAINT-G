# Fase 13 - Modelos Hugging Face Pequenos

Status: **em andamento**.

## Objetivo

Testar SAINT em modelos Hugging Face pequenos, com checkpoints locais e fluxo
compatível com o runtime SAINT.

## Marco 1 - Adaptador Local Dependency-Optional

Status: **concluido**.

Este marco adiciona suporte inicial a modelos Hugging Face sem exigir download
de modelo nem dependencia obrigatoria de `transformers`.

## Implementado

- adapter `huggingface_causal_lm`;
- leitura de state dict JSON local;
- leitura opcional de `.bin`, `.pt` e `.pth` via PyTorch;
- tentativa opcional de `AutoModelForCausalLM.from_pretrained(..., local_files_only=True)`;
- listagem de matrizes 2D por keywords;
- filtro por `max_dim` e `max_matrices`;
- metodo `hf_saint_delta_smoke`;
- deltas SAINT por blocos de maior magnitude;
- checkpoint robusto com dtype/shards;
- resume e merge pelo runtime existente;
- config exemplo `configs/huggingface_smoke.json`;
- testes automatizados sem rede.

## Configuracao

Campos principais:

```text
task: huggingface_causal_lm
method: hf_saint_delta_smoke
metadata.model_name_or_path: caminho local do modelo ou state_dict JSON
metadata.max_dim: recorte maximo por matriz
metadata.max_matrices: numero maximo de matrizes
metadata.keywords: filtros opcionais de nomes de tensores
metadata.checkpoint_dtype: float32 | float16 | bfloat16 | int8
metadata.checkpoint_shard_bytes: limite aproximado por shard
```

## Fluxo Validado

```text
inspect -> train -> checkpoint -> resume -> merge
```

O teste usa um state dict JSON local simulando nomes de camadas Hugging Face:

```text
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.mlp.down_proj.weight
```

## Limites Atuais

- ainda nao mede perplexity real;
- ainda nao usa dataset/tokenizer Hugging Face;
- ainda nao executa autograd real em `transformers`;
- ainda nao compara contra LoRA/QLoRA em modelo Hugging Face real;
- o metodo atual e um smoke de deltas por magnitude, nao treinamento final.

## Marco 2 - Treino Real com Autograd

Status: **concluido**.

Este marco adiciona o caminho `hf_saint_autograd_smoke`, que usa PyTorch
autograd para treinar deltas SAINT sobre matrizes extraidas de um checkpoint
Hugging Face local.

### Entregas

- metodo `hf_saint_autograd_smoke`;
- modulo `saint/adapters/huggingface_autograd.py`;
- selecao de parametros por magnitude;
- deltas treinaveis com PyTorch autograd;
- otimizador AdamW;
- medicao de `initial_loss`;
- medicao de `train_loss`;
- exportacao de `delta_payload`;
- checkpoint robusto com dtype/shards;
- merge avaliavel pelo runtime;
- config exemplo `configs/huggingface_autograd_smoke.json`;
- teste automatizado que executa o fluxo completo quando PyTorch existe;
- erro explicito quando PyTorch nao esta instalado.

### Observacao do Ambiente

No ambiente atual:

```text
torch: 2.11.0+cu128
transformers: 5.8.1
cuda: NVIDIA GeForce RTX 4090
```

### Limites

- o caminho atual usa uma loss proxy sobre deltas de matrizes extraidas;
- ainda nao executa `model.forward` real de `AutoModelForCausalLM`;
- ainda nao mede perplexity com tokenizer/dataset real;
- ainda nao compara contra LoRA/QLoRA.

## Marco 3 - Forward Real Transformers

Status: **concluido**.

Este marco adiciona `hf_saint_forward_smoke`, que carrega um modelo local via
`AutoModelForCausalLM`, carrega tokenizer local, executa `model.forward` real
com `labels`, treina deltas SAINT por autograd e salva checkpoint avaliavel.

### Entregas

- metodo `hf_saint_forward_smoke`;
- modulo `saint/adapters/huggingface_forward.py`;
- carregamento local com `AutoModelForCausalLM.from_pretrained`;
- carregamento local com `AutoTokenizer.from_pretrained`;
- tokenizacao de textos curtos;
- forward real `model(input_ids, labels=input_ids)`;
- aplicacao de deltas por `torch.func.functional_call`;
- selecao de matrizes alvo por keywords;
- medicao de loss inicial;
- medicao de loss final;
- perplexity simples por `exp(loss)`;
- checkpoint robusto com dtype/shards;
- merge dos deltas treinados;
- config exemplo `configs/huggingface_forward_smoke.json`;
- teste com GPT-2 minimo local criado sem rede.

### Fluxo Validado

```text
modelo local -> tokenizer local -> forward real -> treino SAINT -> checkpoint -> merge
```

O teste cria um GPT-2 minimo local com tokenizer `WordLevel`, sem baixar nada da
internet.

## Marco 4 - Comparacao com Baselines HF

Status: **concluido**.

Este marco compara SAINT contra full fine-tuning pequeno no mesmo modelo GPT-2
minimo local, repetindo seeds e medindo throughput, memoria CUDA e checkpoint.

### Entregas

- modulo `saint/adapters/huggingface_benchmark.py`;
- benchmark `benchmark_hf_saint_vs_full`;
- comparacao `hf_saint_forward_smoke` vs `hf_full_finetune`;
- repeticao com seeds `31` e `32`;
- medicao de `tokens_per_s`;
- medicao de `cuda_peak_bytes`;
- contagem de parametros treinaveis;
- checkpoint e merge para SAINT;
- teste automatizado sem rede.

### Resultado CUDA

Configuracao:

```text
modelo: GPT-2 minimo local
device: cuda
seeds: 31, 32
steps: 1
parameter_budget SAINT: 8
```

Resultado:

| metodo | seed | parametros | loss inicial | loss final | delta loss | tokens/s | pico CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAINT | 31 | 8 | 2.792639 | 2.792619 | -0.000021 | 393.51 | 18230784 |
| full | 31 | 3824 | 2.790193 | 2.749064 | -0.041129 | 2915.51 | 18239488 |
| SAINT | 32 | 8 | 2.792639 | 2.792619 | -0.000021 | 5873.83 | 18230784 |
| full | 32 | 3824 | 2.767291 | 2.769696 | 0.002405 | 4872.50 | 18239488 |

Leitura:

- SAINT treinou apenas 8 parametros e reduziu pouco a loss;
- full fine-tuning teve mais capacidade, usando 3824 parametros;
- o checkpoint/merge SAINT passou nas duas seeds;
- a memoria CUDA foi parecida nesse modelo minimo porque o peso base domina o
  custo e o modelo e muito pequeno.

## Proximo Marco

Marco 5 deve melhorar a competicao contra baselines:

- adicionar baseline LoRA no forward real;
- aumentar steps e dataset curto;
- testar modelo pequeno real local, nao apenas GPT-2 minimo sintetico;
- medir qualidade apos `resume`;
- medir ganho por parametro treinavel.
