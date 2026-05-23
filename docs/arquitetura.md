# Arquitetura SAINT-G

SAINT-G significa **DRM por Enxerto com SAINT-G-Phi**.

Este documento descreve a arquitetura baseada no paradigma SAINT-G:

```text
SAINT-G = sparse multi-scale block-codebook delta training
```

Em portugues:

```text
treino de deltas esparsos por dicionario multi-escala de blocos
```

O objetivo do SAINT-G nao e treinar todos os parametros de uma LLM gigante ao mesmo tempo. O objetivo e adaptar ou treinar parcialmente modelos grandes usando:

- loss global;
- atualizacao local;
- deltas esparsos;
- codebooks multi-escala;
- roteamento de blocos;
- orcamento explicito de VRAM;
- recomposicao final.

## 1. Principio Arquitetural

O treino tradicional atualiza muitos parametros simultaneamente.

SAINT-G muda o eixo do problema:

```text
modelo base congelado ou parcialmente congelado
  + deltas pequenos
  + blocos reutilizaveis
  + scheduler de partes
  + loss global
```

O peso efetivo de uma matriz e:

```text
W_eff = W_base + DeltaW
```

Onde:

- `W_base` e a matriz original;
- `DeltaW` e um delta esparso, reconstruido por blocos;
- `W_eff` e a matriz usada no forward.

O modelo inteiro participa do forward e da loss. Mas apenas as partes escolhidas pelo SAINT-G recebem gradiente.

```text
loss global
gradiente local
update parcial
revisao e consolidacao
```

## 2. Visao Geral dos Modulos

Arquitetura logica:

```text
CLI
  -> Config Loader
  -> Model Adapter
  -> Matrix Inspector
  -> Block Partition Engine
  -> Codebook Manager
  -> Sensitivity Analyzer
  -> Block Router
  -> Memory Planner
  -> SAINT-G trainer
  -> Checkpoint Manager
  -> Merger / Recomposer
  -> Evaluator
```

Estrutura prevista:

```text
saint/
  cli/
  config/
  adapters/
  matrices/
  blocks/
  codebooks/
  routing/
  sensitivity/
  memory/
  training/
  checkpoints/
  evaluation/
  merge/
```

## 3. CLI

A CLI e a entrada do usuario.

Comandos previstos:

```bash
SAINT-G inspect --model ./model
SAINT-G reconstruct --matrix ./weights.pt
SAINT-G estimate --model ./model --vram-gb 12
SAINT-G train --config configs/exp.yaml
SAINT-G resume --run runs/exp001
SAINT-G merge --run runs/exp001 --out merged/
SAINT-G compare --run runs/SAINT-G001 --baseline runs/lora001
```

Responsabilidades:

- carregar configuracao;
- validar argumentos;
- iniciar estimativas;
- iniciar treino;
- retomar runs;
- fundir deltas;
- comparar experimentos.

## 4. Configuracao

A configuracao define o contrato do experimento.

Campos principais:

```yaml
project: SAINT-G-exp
model:
  type: hf_causal_lm
  path: ./models/tiny
  base_precision: int4
  train_base: false

data:
  path: ./data/train.txt
  seq_len: 1024

memory:
  vram_gb: 12
  safety_margin_gb: 0.8
  offload: cpu

SAINT-G:
  delta_mode: block_codebook
  block_sizes: [16, 8, 4, 2]
  sparsity_target: 0.01
  codebook_size: 256
  router: heuristic
  sensitivity: grad_norm
  consolidation_interval: 1000

train:
  micro_batch_size: 1
  gradient_accumulation_steps: 32
  learning_rate: 0.0002
  max_steps: 10000
```

## 5. Model Adapter

O Model Adapter padroniza modelos diferentes.

Responsabilidades:

- carregar modelo;
- listar camadas;
- listar matrizes treinaveis;
- expor forward;
- congelar parametros base;
- aplicar deltas SAINT-G;
- remover deltas;
- salvar/carregar estado.

Interface conceitual:

```text
load_model()
list_layers()
list_matrices()
freeze_base()
attach_delta(matrix_id, delta_module)
forward(batch)
state_dict_trainable()
```

Adapters iniciais:

- `DRMTransformerAdapter`;
- `HFCausalLMAdapter`;
- `LinearToyAdapter`;
- `MiniTransformerAdapter`.

## 6. Matrix Inspector

O Matrix Inspector encontra matrizes dentro do modelo.

Ele classifica:

- embeddings;
- attention query;
- attention key;
- attention value;
- attention output;
- MLP up;
- MLP gate;
- MLP down;
- lm head;
- outras matrizes.

Para cada matriz, registra:

```text
id
nome
camada
tipo
shape
dtype
device
numero de parametros
estimativa de custo
```

Exemplo:

```text
layer.12.attention.wq
shape: 4096 x 4096
tipo: attention_q
parametros: 16.7M
```

## 7. Block Partition Engine

Este modulo divide matrizes em regioes e blocos.

Suporta:

- blocos `2x2`;
- blocos `3x3`;
- blocos `4x4`;
- blocos `5x5`;
- blocos `6x6`;
- blocos `8x8`;
- blocos `16x16`;
- blocos maiores para busca inicial;
- padding;
- blocos de borda;
- particionamento hierarquico.

Fluxo:

```text
W
  -> regioes grandes
  -> sub-regioes
  -> blocos
  -> assinaturas
  -> grupos
```

A divisao nao e o ganho por si so. O ganho vem de:

- esparsidade;
- compartilhamento;
- codebook;
- roteamento;
- treino local.

## 8. Assinaturas de Blocos

Cada bloco pode receber uma assinatura.

Assinaturas possiveis:

- valores quantizados;
- norma;
- traco;
- determinante;
- autovalores;
- estatisticas simples;
- hash aproximado;
- cluster;
- id de codebook.

Uso:

- encontrar blocos iguais;
- encontrar blocos parecidos;
- agrupar calculos;
- inicializar codebooks;
- medir reutilizacao.

Observacao: determinante nao reconstrui o bloco sozinho. Ele e apenas uma estatistica possivel.

## 9. Codebook Manager

O Codebook Manager administra dicionarios de blocos.

SAINT-G usa codebooks multi-escala:

```text
codebook_2x2
codebook_3x3
codebook_4x4
codebook_5x5
codebook_6x6
codebook_8x8
codebook_16x16
```

Cada entrada e um prototipo treinavel ou fixo.

Representacao simples:

```text
Dij = escala_ij * codebook_k[id_ij]
```

Representacao por mistura:

```text
Dij = alpha1 * codebook_k[a]
    + alpha2 * codebook_k[b]
    + alpha3 * codebook_k[c]
```

Representacao multi-escala:

```text
D_region = bloco_16x16
         + refinamento_8x8
         + refinamento_4x4
         + refinamento_2x2
```

Responsabilidades:

- criar codebooks;
- inicializar por clustering;
- inicializar aleatoriamente;
- agrupar blocos similares;
- manter indices;
- manter escalas;
- aplicar prototipos;
- acumular gradientes compartilhados;
- salvar e carregar codebooks.

## 10. Delta Representation

O delta SAINT-G e a estrutura treinavel anexada a uma matriz base.

Pode conter:

- mascara esparsa;
- ids de codebook;
- escalas por bloco;
- misturas;
- refinamentos multi-escala;
- blocos livres em regioes criticas;
- LoRA auxiliar opcional.

Forma conceitual:

```text
DeltaW = M * reconstruct(codebooks, ids, scales, refinements)
```

Onde:

- `M` e a mascara esparsa;
- `codebooks` contem prototipos;
- `ids` apontam para prototipos;
- `scales` ajustam intensidade;
- `refinements` adicionam detalhes locais.

## 11. Sensitivity Analyzer

O Sensitivity Analyzer mede quais partes importam.

Entradas:

- modelo;
- batches de amostra;
- loss;
- gradientes;
- ativacoes;
- matriz inspecionada;
- estado atual dos deltas.

Metricas:

- norma do gradiente;
- gradiente vezes peso;
- impacto de mascaramento;
- estimativa Fisher;
- magnitude de ativacao;
- erro de reconstrucao;
- frequencia de padrao;
- ganho por byte.

Saida:

```text
ranking de camadas
ranking de matrizes
ranking de regioes
ranking de blocos
ranking de codebook entries
```

O objetivo e evitar treinar blocos irrelevantes.

## 12. Block Router

O Block Router decide como cada regiao sera representada.

Opcoes:

```text
congelar
usar bloco 16x16
usar bloco 8x8
usar bloco 4x4
usar bloco 2x2
usar mistura multi-escala
usar delta livre
usar LoRA auxiliar
```

Politica inicial:

```text
erro baixo + baixa sensibilidade  -> congelar ou bloco grande
erro alto + baixa sensibilidade   -> bloco medio
erro baixo + alta sensibilidade   -> bloco pequeno
erro alto + alta sensibilidade    -> bloco pequeno + LoRA/delta livre
```

O roteador deve respeitar:

- orcamento de VRAM;
- orcamento por camada;
- limite de parametros treinaveis;
- limite de codebook entries;
- custo de offload;
- estabilidade do treino.

## 13. Memory Planner

O Memory Planner transforma `--vram-gb` em plano executavel.

Ele estima:

- pesos base em GPU;
- pesos base em CPU;
- deltas treinaveis;
- gradientes dos deltas;
- estados do otimizador;
- ativacoes;
- caches;
- buffers temporarios;
- margem de seguranca.

Tambem decide:

- micro-batch;
- seq_len;
- gradient accumulation;
- precision;
- offload;
- numero maximo de blocos ativos;
- tamanho maximo de codebook ativo;
- quantas matrizes podem ser treinadas por fase.

Exemplo de saida:

```text
VRAM alvo: 12.0 GB
Margem: 0.8 GB
Modelo base: int4 + offload CPU
Parametros treinaveis: 18.2M
Blocos ativos: 0.7%
Codebook entries ativas: 512
Micro-batch: 1
Seq len: 1024
Status: viavel
```

## 14. Orcamento por Camada

SAINT-G nao distribui memoria igualmente.

Cada camada e tipo de matriz recebe um orcamento.

Exemplo:

```text
layer.0: 2%
layer.8: 6%
layer.16: 10%
layer.24: 4%
layer.final: 8%
```

Por tipo:

```text
attention.Wq: 10%
attention.Wk: 5%
attention.Wv: 15%
attention.Wo: 10%
mlp.up: 25%
mlp.gate: 20%
mlp.down: 15%
```

Esse orcamento pode ser fixo ou adaptativo.

## 15. SAINT-G trainer

O trainer executa:

```text
forward do modelo inteiro
loss global
backward
gradiente apenas nos deltas ativos
optimizer step local
checkpoint parcial
```

Responsabilidades:

- mixed precision;
- gradient accumulation;
- gradient clipping;
- logging;
- consolidacao;
- avaliacao periodica;
- OOM recovery;
- resume.

O trainer nao deve conhecer detalhes de cada arquitetura. Ele usa o `ModelAdapter`.

## 16. Scheduler de Fases

O Scheduler define a ordem do treino.

Estrategias:

- sequencial por camada;
- reversa;
- por sensibilidade;
- por ganho por byte;
- por erro de reconstrucao;
- por reuso de padrao;
- curriculum por tamanho;
- aleatoria controlada.

Exemplo:

```text
fase 1: lm_head e ultimas camadas
fase 2: MLPs mais sensiveis
fase 3: attention.Wv com alto ganho por byte
fase 4: refinamento 4x4
fase 5: refinamento 2x2 em regioes criticas
fase 6: consolidacao
```

## 17. Consolidacao

Treinar partes separadas pode gerar conflito.

Consolidacao e uma fase curta onde SAINT-G revisita partes importantes com os deltas ja ativos.

Objetivos:

- reduzir conflito entre deltas;
- recuperar qualidade global;
- estabilizar recomposicao;
- medir degradacao;
- recalcular sensibilidade.

Exemplo:

```text
treina grupo A
treina grupo B
treina grupo C
consolidacao A+B+C
```

## 18. Cache de Grupos

Se varios blocos compartilham o mesmo prototipo, SAINT-G pode agrupar calculos.

Exemplo:

```text
grupo G7:
  codebook_id = 7
  posicoes = [D12, D98, D301]
```

Runtime:

```text
calcular prototipo uma vez
aplicar em varias posicoes
acumular gradientes no mesmo prototipo
```

Isso so ajuda se houver repeticao real ou repeticao induzida pelo codebook.

## 19. Checkpoint Manager

Checkpoint SAINT-G nao precisa salvar o modelo inteiro.

Pode salvar:

- config efetiva;
- plano de memoria;
- mapa de sensibilidade;
- roteador;
- codebooks;
- ids de blocos;
- escalas;
- mascaras;
- deltas livres;
- LoRA auxiliar;
- estado do otimizador ativo;
- scheduler;
- logs.

Estrutura:

```text
runs/
  exp001/
    config.yaml
    plan.json
    logs.jsonl
    sensitivity.json
    router.json
    codebooks/
    checkpoints/
      step_000100/
        trainable.pt
        optimizer.pt
        scheduler.json
    merged/
```

## 20. Merger / Recomposer

O Merger transforma deltas em modelo utilizavel.

Modos:

### Composto

Mantem modelo base + deltas.

```text
W_eff = W_base + reconstruct(DeltaW)
```

Vantagens:

- checkpoint pequeno;
- reversivel;
- permite trocar deltas.

### Fundido

Materializa pesos finais.

```text
W_final = W_base + DeltaW
```

Vantagens:

- inferencia mais simples;
- nao precisa reconstruir em tempo real.

Desvantagens:

- checkpoint maior;
- perde modularidade.

## 21. Evaluator

O Evaluator compara SAINT-G contra baselines.

Baselines:

- full fine-tuning em modelo pequeno;
- LoRA;
- QLoRA;
- adapters;
- head-only;
- bloco sem codebook;
- codebook tamanho unico;
- codebook multi-escala.

Metricas:

- loss;
- perplexity;
- tokens/s;
- VRAM maxima;
- RAM;
- parametros treinaveis;
- tamanho de checkpoint;
- taxa de reutilizacao;
- erro de reconstrucao;
- ganho por byte;
- degradacao apos merge.

## 22. Data Pipeline

O Data Pipeline deve ser simples e streaming.

Formatos:

- `.txt`;
- `.jsonl`;
- `.npy`;
- `.bin`;
- shards tokenizados.

Responsabilidades:

- carregar sem estourar RAM;
- gerar janelas;
- controlar shuffle;
- salvar metadados;
- permitir datasets pequenos para testes.

## 23. Politica de OOM

Se ocorrer OOM:

1. limpar cache CUDA;
2. reduzir blocos ativos;
3. reduzir codebook entries ativas;
4. reduzir micro-batch;
5. reduzir seq_len se permitido;
6. aumentar gradient accumulation;
7. aumentar offload;
8. trocar blocos menores por representacao maior/mais compacta;
9. abortar com relatorio se continuar inviavel.

SAINT-G nao deve entrar em loop infinito tentando configuracoes aleatorias.

## 24. Fluxo de Treino

Fluxo completo:

```text
1. carregar config
2. carregar modelo base
3. inspecionar matrizes
4. estimar memoria
5. particionar matrizes candidatas
6. criar assinaturas
7. criar ou carregar codebooks
8. medir reconstrucao
9. medir sensibilidade
10. rotear regioes
11. criar plano de fases
12. treinar deltas ativos com loss global
13. salvar checkpoint
14. consolidar
15. avaliar
16. recompor/fundir
```

## 25. Fluxo de Reconstrucao de Matriz

Antes de LLM, SAINT-G precisa provar reconstrucao.

```text
W
  -> particionar
  -> agrupar blocos
  -> criar codebook
  -> reconstruir W_aprox
  -> medir erro
  -> comparar com SVD/LoRA/quantizacao
```

Esse fluxo valida a representacao sem misturar treino de modelo.

## 26. Integracao com drm_transformer

O `drm_transformer` entra como primeiro adapter customizado.

O adapter deve:

- carregar `DRMTransformer`;
- listar blocos e matrizes;
- congelar base;
- anexar deltas SAINT-G;
- executar forward;
- expor loss;
- salvar deltas;
- comparar contra trainer original.

Arquitetura:

```text
SAINT-G trainer
  -> DRMTransformerAdapter
      -> DRMTransformer
      -> matrizes
      -> deltas SAINT-G
```

## 27. Escalabilidade

SAINT-G deve escalar por etapas:

```text
matriz isolada
camada linear
mini-transformer
drm_transformer pequeno
modelo HF pequeno
3B
14B
70B
```

Nao escalar antes de provar.

## 28. Decisoes de Design

1. O modelo base deve permanecer congelado por padrao.
2. Deltas sao o principal objeto de treino.
3. Codebooks devem ser multi-escala.
4. Roteamento deve ser explicavel antes de ser treinavel.
5. Sensibilidade deve guiar escolha de partes.
6. Orcamento de VRAM e restricao central, nao detalhe.
7. Reconstrucao de matriz deve ser testada antes de LLM.
8. Baselines LoRA/QLoRA sao obrigatorias.
9. Checkpoints devem ser recomponiveis.
10. O sistema deve mostrar quando uma meta e inviavel.

## 29. Criterios de Sucesso

SAINT-G sera promissor se mostrar pelo menos uma vantagem:

- menor memoria que LoRA em algum regime;
- menor checkpoint;
- melhor ganho por parametro treinavel;
- melhor ganho por byte;
- boa adaptacao com base congelada;
- codebook reutilizavel;
- recomposicao estavel.

SAINT-G sera fraco se:

- nao comprimir matrizes melhor que alternativas simples;
- nao convergir em modelos pequenos;
- perder sempre para LoRA/QLoRA;
- tiver overhead alto demais;
- nao reutilizar padroes;
- depender de ajustes manuais excessivos.

## 30. Resumo

A arquitetura SAINT-G e um runtime para:

```text
inspecionar matrizes
particionar em blocos multi-escala
criar codebooks
medir sensibilidade
rotear regioes
treinar deltas esparsos
consolidar
recompor modelo final
```

O nucleo e:

```text
loss global,
atualizacao local,
padroes compartilhados,
calculo agrupado,
recomposicao final
```
