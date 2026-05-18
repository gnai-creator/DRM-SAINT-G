# SAINT - Arquitetura

SAINT significa **Simple AI Node Training**. O objetivo do projeto e criar uma camada simples para treinar, adaptar e experimentar modelos de IA em computadores comuns, respeitando um limite explicito de VRAM definido pelo usuario.

O foco inicial nao e prometer treino completo de modelos gigantes em uma unica GPU domestica. Um modelo de 20B, 70B ou 640B parametros nao cabe integralmente em 12GB ou 24GB de VRAM quando consideramos pesos, gradientes, estados do otimizador e ativacoes. O caminho viavel e transformar SAINT em um sistema de **treino parcial, fine-tuning, adapters, LoRA, blocos congelados/descongelados, offload e execucao em micro-blocos**.

## Objetivos

1. Permitir que o usuario informe quanta VRAM deseja usar.
2. Dividir o treino em unidades pequenas e controlaveis.
3. Suportar modelos tipo LLM e, no futuro, world models.
4. Suportar treino parcial de modelos grandes usando tecnicas de baixo consumo.
5. Expor uma CLI simples para experimentos locais.
6. Usar o `drm_transformer` como primeiro exemplo de modelo customizado.

## Nao Objetivos

1. Treinar todos os parametros de um modelo 20B, 70B ou 640B do zero em uma RTX 3060/4090.
2. Esconder limites fisicos de memoria, largura de banda e tempo.
3. Criar um framework distribuido completo antes de validar o fluxo local.
4. Substituir PyTorch, Hugging Face, DeepSpeed ou FSDP; SAINT deve orquestrar e simplificar.

## Principio Central

O usuario declara um orcamento:

```bash
SAINT train --model drm_transformer --data ./data --vram-gb 12
```

SAINT converte esse orcamento em uma estrategia:

- tamanho de micro-batch;
- comprimento de sequencia;
- precision (`fp32`, `fp16`, `bf16`, quantizado);
- gradient accumulation;
- gradient checkpointing;
- quais parametros ficam treinaveis;
- quais blocos ficam em GPU, CPU ou disco;
- frequencia de checkpoint;
- estimativa de memoria antes do treino.

## Camadas da Arquitetura

### 1. CLI

Entrada principal do usuario.

Responsabilidades:

- ler argumentos;
- carregar arquivo de configuracao opcional;
- validar limites de memoria;
- iniciar treino, avaliacao ou estimativa;
- imprimir plano de execucao antes de iniciar.

Comandos previstos:

```bash
SAINT estimate --model ./model --vram-gb 12
SAINT train --config configs/local_12gb.yaml
SAINT resume --checkpoint runs/exp001/latest
SAINT inspect --checkpoint runs/exp001/latest
```

### 2. Configuracao

Representa o contrato entre usuario e runtime.

Campos principais:

- `model_type`: `drm_transformer`, `hf_causal_lm`, `world_model`;
- `model_path`: caminho local ou identificador;
- `data_path`: caminho dos dados;
- `vram_gb`: limite de VRAM desejado;
- `train_mode`: `full_small`, `lora`, `adapter`, `blockwise`, `head_only`;
- `precision`: `auto`, `fp32`, `fp16`, `bf16`, `int8`, `int4`;
- `seq_len`;
- `micro_batch_size`;
- `gradient_accumulation_steps`;
- `checkpointing`;
- `offload`: `none`, `cpu`, `nvme`;
- `target_modules`: lista de blocos ou modulos treinaveis.

### 3. Memory Planner

Componente mais importante do SAINT.

Responsabilidades:

- estimar memoria dos pesos;
- estimar memoria dos gradientes;
- estimar estados do otimizador;
- estimar ativacoes por batch e sequencia;
- calcular margem de seguranca;
- rejeitar planos impossiveis;
- sugerir plano alternativo.

Exemplo de saida:

```text
VRAM alvo: 12.0 GB
Modelo: 7.0B parametros
Modo: LoRA
Precision: int4 base + bf16 adapters
Micro-batch: 1
Seq len: 1024
Gradient accumulation: 32
Gradient checkpointing: on
Offload: cpu
Status: viavel com margem estimada de 1.4 GB
```

### 4. Model Adapter

Camada que padroniza modelos diferentes.

Interface esperada:

- carregar modelo;
- listar blocos;
- congelar parametros;
- ativar parametros selecionados;
- inserir LoRA/adapters;
- mover blocos entre devices;
- executar forward;
- salvar pesos treinaveis;
- carregar checkpoint parcial.

Primeiros adapters:

1. `DRMTransformerAdapter`
2. `HFCausalLMAdapter`
3. `WorldModelAdapter` futuro

### 5. Training Modes

SAINT deve suportar varios modos de treino.

#### `full_small`

Treina todos os parametros.

Uso:

- modelos pequenos;
- testes;
- validacao do pipeline.

#### `head_only`

Treina apenas cabeca de saida ou pequenos modulos finais.

Uso:

- classificacao;
- adaptacao barata;
- smoke tests.

#### `lora`

Congela o modelo base e treina matrizes LoRA.

Uso:

- principal modo para modelos grandes;
- ideal para GPUs domesticas.

#### `adapter`

Insere pequenos modulos treinaveis entre blocos.

Uso:

- adaptacao persistente;
- experimentos com world models.

#### `blockwise`

Treina um subconjunto de blocos por vez.

Uso:

- pesquisa;
- modelos que nao cabem com todos os blocos ativos;
- ajuste progressivo.

Limitacao importante: treino por blocos nao e equivalente a treino full end-to-end. Ele pode funcionar como aproximacao, adaptacao ou pretreino faseado, mas deve ser medido empiricamente.

### 6. Block Scheduler

Controla quais partes do modelo estao ativas em cada etapa.

Exemplo:

```text
fase 1: embeddings + blocos 0-1
fase 2: blocos 2-3
fase 3: blocos 4-5
fase 4: lm_head
fase 5: adapters globais
```

Responsabilidades:

- selecionar blocos treinaveis;
- congelar o restante;
- mover blocos para GPU sob demanda;
- aplicar accumulation;
- sincronizar checkpoints parciais;
- evitar esquecimento catastrofico com fases de revisao.

### 7. Data Pipeline

Pipeline simples e streaming.

Responsabilidades:

- ler texto, tokens ou shards;
- evitar carregar dataset inteiro em RAM;
- gerar janelas de treino;
- suportar shuffle controlado;
- salvar metadados de tokenizacao;
- permitir datasets pequenos para testes.

Formatos iniciais:

- `.txt`;
- `.jsonl`;
- `.npy`;
- `.bin`;
- shards tokenizados.

### 8. Trainer

Executa o loop de treino.

Responsabilidades:

- mixed precision;
- gradient accumulation;
- clipping;
- checkpointing;
- logging;
- avaliacao periodica;
- resume;
- metricas de tokens/s, loss e memoria.

O trainer nao deve conhecer detalhes internos de cada modelo. Ele conversa com `ModelAdapter`.

### 9. Checkpoint Manager

Responsabilidades:

- salvar checkpoint completo para modelos pequenos;
- salvar apenas adapters/LoRA quando aplicavel;
- salvar estado do scheduler;
- salvar estado do otimizador;
- salvar configuracao efetiva;
- salvar historico de treino;
- permitir resume exato.

Estrutura prevista:

```text
runs/
  exp001/
    config.yaml
    plan.json
    logs.jsonl
    checkpoints/
      step_000100/
        trainable.pt
        optimizer.pt
        scheduler.json
      latest/
```

### 10. Observabilidade

SAINT deve mostrar ao usuario o que esta acontecendo.

Metricas:

- VRAM alocada;
- VRAM reservada;
- RAM usada;
- tokens por segundo;
- loss;
- grad norm;
- learning rate;
- parametros totais;
- parametros treinaveis;
- porcentagem treinavel;
- tempo estimado por etapa.

## Fluxo de Treino

1. Usuario chama `SAINT train`.
2. CLI carrega config.
3. Model Adapter inspeciona modelo.
4. Memory Planner cria plano de execucao.
5. SAINT imprime o plano e avisos.
6. Data Pipeline prepara batches.
7. Block Scheduler define parametros ativos.
8. Trainer executa micro-steps.
9. Checkpoint Manager salva progresso.
10. Avaliador mede loss ou metricas especificas.
11. Runtime ajusta plano se houver OOM.

## Politica de OOM

Se ocorrer out-of-memory:

1. limpar cache CUDA;
2. reduzir micro-batch;
3. reduzir seq_len se permitido;
4. aumentar gradient accumulation;
5. ativar checkpointing se desligado;
6. aumentar offload;
7. se ainda falhar, abortar com plano sugerido.

SAINT nao deve entrar em loop infinito tentando configuracoes aleatorias.

## Integracao com `drm_transformer`

O `drm_transformer` pode ser o primeiro modelo usado pelo SAINT porque ja possui:

- configuracao propria;
- modelo PyTorch;
- trainer;
- dataset por shards;
- suporte a mixed precision;
- checkpointing;
- metricas especificas.

No SAINT, ele deve entrar por um adapter:

```text
SAINT Trainer
  -> DRMTransformerAdapter
      -> DRMTransformer
      -> DRMTransformerConfig
```

O adapter deve traduzir chamadas genericas como `freeze_all`, `enable_block`, `save_trainable` e `forward` para a API real do `drm_transformer`.

## Realidade Sobre Modelos Gigantes

Estimativa simplificada apenas para pesos:

```text
20B em fp16  ~= 40 GB so pesos
70B em fp16  ~= 140 GB so pesos
640B em fp16 ~= 1280 GB so pesos
```

Treino completo normalmente exige muito mais memoria por causa de:

- gradientes;
- Adam moments;
- ativacoes;
- buffers temporarios;
- fragmentacao;
- contexto longo.

Portanto, em GPUs de 12GB ou 24GB, SAINT deve usar:

- quantizacao para o modelo base;
- LoRA/adapters para parametros treinaveis;
- offload CPU/NVMe;
- treino por blocos;
- micro-batch pequeno;
- gradient accumulation.

## Roadmap Inicial

### Fase 1 - Documento e Esqueleto

- definir arquitetura;
- criar pacote `SAINT`;
- criar CLI minima;
- criar config dataclass;
- criar memory planner estimativo;
- criar testes unitarios do planner.

### Fase 2 - Treino Pequeno

- implementar treino `full_small`;
- treinar modelo pequeno local;
- salvar checkpoints;
- validar resume.

### Fase 3 - Adapter DRM

- integrar `drm_transformer`;
- listar blocos;
- congelar/descongelar parametros;
- executar treino simples;
- medir memoria.

### Fase 4 - LoRA/Adapters

- inserir LoRA em camadas lineares;
- salvar somente pesos LoRA;
- carregar LoRA sobre modelo base;
- estimar memoria treinavel.

### Fase 5 - Blockwise

- scheduler de blocos;
- offload CPU;
- checkpoints parciais;
- politicas de revisao.

### Fase 6 - Modelos Externos

- adapter Hugging Face;
- suporte a modelos quantizados;
- streaming datasets;
- avaliacao padronizada.

## Criterios de Sucesso

O primeiro SAINT funcional deve conseguir:

1. receber `--vram-gb`;
2. estimar se um plano cabe;
3. treinar um modelo pequeno de ponta a ponta;
4. treinar somente parametros selecionados de um modelo maior;
5. salvar e retomar checkpoints;
6. mostrar claramente quando uma meta e inviavel.

## Decisao de Design

SAINT deve ser honesto e simples. O valor do projeto nao esta em fingir que uma GPU domestica treina um 640B completo, mas em tornar experimentos reais de IA mais acessiveis usando tecnicas de memoria limitada de forma automatizada e compreensivel.
