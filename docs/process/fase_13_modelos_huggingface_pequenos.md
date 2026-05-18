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

Status: **implementado, pendente de execucao com PyTorch/Transformers no ambiente atual**.

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
torch: ausente
transformers: ausente
```

Por isso, o teste valida o contrato de dependencia. Em um ambiente com PyTorch,
o mesmo teste executa:

```text
train -> checkpoint -> merge
```

e verifica se a loss final nao piora em relacao a `initial_loss`.

### Limites

- o caminho atual usa uma loss proxy sobre deltas de matrizes extraidas;
- ainda nao executa `model.forward` real de `AutoModelForCausalLM`;
- ainda nao mede perplexity com tokenizer/dataset real;
- ainda nao compara contra LoRA/QLoRA.

## Proximo Marco

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

## Proximo Marco

Marco 4 deve comparar SAINT contra baselines no mesmo modelo:

- comparar contra LoRA ou full fine-tuning pequeno;
- repetir com seeds diferentes;
- medir memoria CUDA;
- medir tokens/s;
- registrar checkpoint e merge avaliavel.
