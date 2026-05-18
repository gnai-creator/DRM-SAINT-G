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

## Marco 5 - LoRA e Modelo HF Real Local

Status: **concluido**.

Este marco melhora a comparacao contra baselines no mesmo forward real de
`AutoModelForCausalLM`.

### Entregas

- baseline LoRA sem dependencia de `peft`;
- aplicacao LoRA por `torch.func.functional_call`;
- dataset curto ampliado para seis textos;
- benchmark com mais steps no teste automatizado;
- medicao de `resume_quality_delta` apos `resume`;
- medicao de `gain_per_parameter`;
- comparacao SAINT vs LoRA vs full fine-tuning;
- suporte ao mesmo caminho `model_name_or_path` local para testar modelos HF
  reais ja presentes na maquina.

### Baselines

```text
hf_saint_forward_smoke
hf_lora_rank_2
hf_full_finetune
```

### Observacao

O teste automatizado continua criando um GPT-2 minimo local sem rede. Isso
garante reproducibilidade no CI e em ambientes sem modelos baixados. Para testar
um modelo pequeno real local, use o mesmo benchmark apontando `model_path` para
um diretorio Hugging Face ja existente na maquina.

### Metricas Adicionadas

- `resume_train_loss`;
- `resume_quality_delta`;
- `gain_per_parameter`;
- `tokens_per_s`;
- `cuda_peak_bytes`.

### Resultado CUDA Smoke

Configuracao:

```text
modelo: GPT-2 minimo local
device: cuda
seed: 31
steps: 2
parameter_budget SAINT: 8
LoRA rank: 2
```

Resultado:

| metodo | parametros | loss inicial | loss final | delta loss | ganho/param | tokens/s | pico CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAINT | 8 | 3.425533 | 3.425443 | -0.000090 | 0.00001124 | 647.89 | 18240000 |
| LoRA r2 | 192 | 3.432518 | 3.432486 | -0.000032 | 0.00000017 | 9403.80 | 18340352 |
| full | 4064 | 3.435688 | 3.386605 | -0.049083 | 0.00001208 | 6108.98 | 18395136 |

O `resume_quality_delta` do SAINT foi `0.0`, confirmando que a qualidade
registrada no checkpoint foi preservada apos `resume`.

### Leitura Tecnica

O Marco 5 ainda nao prova vantagem contra LoRA. Ele fecha a lacuna de
instrumentacao: agora SAINT, LoRA e full fine-tuning rodam no mesmo forward real,
com o mesmo dataset curto e metricas comparaveis de qualidade, throughput,
memoria e eficiencia por parametro.

## Proximo Marco

## Marco 6 - Sweep HF Real Local

Status: **concluido**.

Este marco transforma a comparacao do Marco 5 em sweep reproduzivel.

### Entregas

- modulo `saint/adapters/huggingface_sweep.py`;
- script `scripts/benchmark_huggingface_phase13.py`;
- sweep de budgets SAINT;
- sweep de LoRA ranks `1`, `2`, `4` e `8`;
- exportacao de `results.json`;
- exportacao de `results.md`;
- medicao de perplexity inicial, de treino e apos merge;
- medicao de memoria CUDA em runs mais longas;
- execucao em modelo HF pequeno real local.

### Modelo Local Real

Foi usado:

```text
sshleifer/tiny-gpt2
```

O modelo foi baixado uma vez e salvo localmente em:

```text
models/sshleifer_tiny_gpt2
```

A pasta `models/` e ignorada pelo Git.

### Comando

```bash
python scripts/benchmark_huggingface_phase13.py \
  --model models/sshleifer_tiny_gpt2 \
  --out runs/phase13_marco6_sweep \
  --device cuda \
  --steps 6 \
  --saint-budgets 4,8,16 \
  --lora-ranks 1,2,4,8
```

### Resultado CUDA

| metodo | budget | rank | parametros | loss final | perplexity apos merge | ganho/param | tokens/s | pico CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAINT | 4 |  | 4 | 10.824375 | 49923.772119 | 0.00000119 | 2201.15 | 31205888 |
| SAINT | 8 |  | 8 | 10.824288 | 49919.201671 | 0.00001144 | 5283.24 | 31205888 |
| SAINT | 16 |  | 12 | 10.824239 | 49916.583373 | 0.00001176 | 5214.95 | 31205888 |
| LoRA |  | 1 | 12 | 10.818256 | 49923.962564 | 0.00000008 | 9722.52 | 43878400 |
| LoRA |  | 2 | 24 | 10.818254 | 49923.867341 | 0.00000012 | 7673.50 | 43878400 |
| LoRA |  | 4 | 48 | 10.818251 | 49923.676897 | 0.00000014 | 9589.04 | 43878400 |
| LoRA |  | 8 | 96 | 10.818245 | 49923.391233 | 0.00000013 | 10402.43 | 43878400 |
| full |  |  | 102714 | 10.806647 | 49347.742578 | 0.00000012 | 8155.16 | 44737024 |

### Leitura Tecnica

- full fine-tuning ainda reduz mais loss absoluta;
- LoRA reduz loss com mais throughput, mas usa mais parametros que SAINT nos
  budgets testados;
- SAINT teve melhor ganho por parametro que LoRA neste sweep curto;
- SAINT usou menos pico CUDA que LoRA e full neste modelo;
- a perplexity apos merge foi registrada para validar que o checkpoint
  recomponivel continua avaliavel.

## Proximo Marco

Marco 7 deve aproximar a tarefa de um fine-tuning real:

- usar dataset textual externo pequeno;
- separar treino e validacao;
- acumular gradiente em mais batches;
- testar learning rates diferentes para SAINT e LoRA;
- salvar checkpoint SAINT e adaptador LoRA em formatos comparaveis;
- medir qualidade de geracao curta antes e depois do merge.
