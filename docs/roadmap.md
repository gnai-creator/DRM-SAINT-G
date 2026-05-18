# Roadmap SAINT

SAINT significa **Simple AI Node Training**. Este roadmap organiza o desenvolvimento do paradigma em fases verificaveis, partindo de experimentos matematicos pequenos ate testes em modelos grandes.

O objetivo nao e comecar pelo 70B. O objetivo e provar, passo a passo, que o paradigma funciona:

```text
matriz isolada
  -> camada linear
  -> mini-transformer
  -> modelo pequeno
  -> modelo 3B
  -> modelo 14B
  -> modelo 70B
```

O principio do SAINT e:

```text
loss global,
atualizacao local,
padroes compartilhados,
calculo agrupado,
recomposicao final
```

## Status Atual

```text
Fase atual: Fase 1 - Biblioteca de Blocos
Fase anterior: Fase 0 concluida
Proximo marco: Marco A - Prova de Matriz
```

Resumo do estado:

| Fase | Nome | Status |
|---|---|---|
| 0 | Fundacao Conceitual | Concluida |
| 1 | Biblioteca de Blocos | Proxima |
| 2 | Benchmark de Reconstrucao | Pendente |
| 3 | Roteador de Blocos | Pendente |
| 4 | Treino de Camada Linear | Pendente |
| 5 | Mini-Transformer | Pendente |
| 6+ | Escala e runtime completo | Pendente |

## 1. Fase 0 - Fundacao Conceitual

Status: **concluida**.

### Objetivo

Consolidar o paradigma antes de escrever um runtime grande.

### Entregas

- `docs/paradigma_treino_tradicional.md`
- `docs/paradigma_SAINT.md`
- `docs/arquitetura.md`
- `docs/roadmap.md`
- `docs/glossario_tecnico.md`
- `docs/hipoteses.md`
- `docs/criterios_sucesso_falha.md`

### Perguntas

- O que SAINT tenta melhorar em relacao ao treino tradicional?
- O que SAINT nao promete?
- Quais baselines serao usados?
- Quais experimentos invalidam a ideia?

### Criterio de conclusao

A fase termina quando o projeto tem uma definicao clara:

```text
SAINT = sparse multi-scale block-codebook delta training
```

Em portugues:

```text
treino de deltas esparsos por dicionario multi-escala de blocos
```

### Resultado

A Fase 0 estabeleceu:

- definicao tecnica do paradigma;
- comparacao com treino tradicional;
- arquitetura baseada em deltas esparsos e codebook multi-escala;
- hipoteses testaveis;
- criterios objetivos de sucesso e falha;
- glossario comum para os proximos documentos e implementacoes.

### Decisao

```text
continuar para Fase 1 - Biblioteca de Blocos
```

Motivo:

```text
o paradigma esta definido o suficiente para iniciar validacao matematica
sem escrever ainda um runtime grande.
```

## 2. Fase 1 - Biblioteca de Blocos

Status: **proxima fase**.

### Objetivo

Criar operacoes basicas para dividir, agrupar, reconstruir e medir matrizes.

### Componentes

- particionador de matrizes em blocos;
- suporte a blocos `2x2`, `3x3`, `4x4`, `5x5`, `6x6`, `8x8`, `16x16`;
- padding para dimensoes nao divisiveis;
- reconstrutor de matriz;
- calculo de erro de reconstrucao;
- assinaturas de blocos;
- agrupamento de blocos iguais ou parecidos;
- codebook fixo;
- codebook treinavel.

### Entregas Tecnicas

- pacote inicial `saint`;
- modulo `blocks`;
- funcao para particionar matrizes;
- funcao para reconstruir matrizes;
- suporte a padding;
- suporte a blocos de borda;
- calculo de assinaturas;
- agrupamento exato;
- agrupamento aproximado por quantizacao;
- testes unitarios.

### API Inicial Desejada

```python
blocks = partition_matrix(W, block_size=(4, 4))
W_recon = reconstruct_matrix(blocks, original_shape=W.shape)
signatures = compute_block_signatures(blocks)
groups = group_blocks(signatures)
```

### Experimentos

1. Dividir e reconstruir matriz sem perda.
2. Dividir matriz com padding e reconstruir dimensao original.
3. Agrupar blocos exatamente iguais.
4. Agrupar blocos aproximados por quantizacao.
5. Medir erro de reconstrucao por tamanho de bloco.

### Metricas

- erro L1;
- erro L2;
- erro relativo;
- compressao;
- numero de blocos;
- numero de prototipos;
- taxa de reutilizacao;
- tempo de particionamento;
- tempo de reconstrucao.

### Criterio de conclusao

A fase termina quando SAINT consegue:

```text
W -> blocos -> codebook -> W_aprox
```

e reporta erro, compressao e taxa de reutilizacao.

### Criterio de Falha

Reavaliar a fase se:

- particionar e reconstruir sem perda for instavel;
- padding gerar erros de shape;
- agrupamento aproximado destruir informacao demais;
- a API ficar acoplada demais a LLMs antes da hora.

## 3. Fase 2 - Benchmark de Reconstrucao

Status: **pendente**.

### Objetivo

Testar se o codebook multi-escala representa matrizes melhor que alternativas simples.

### Baselines

- matriz original sem compressao;
- SVD truncado;
- LoRA equivalente;
- quantizacao uniforme;
- blocos fixos sem codebook;
- codebook unico `2x2`;
- codebook unico `4x4`;
- codebook multi-escala.

### Matrizes de Teste

- matrizes aleatorias gaussianas;
- matrizes low-rank;
- matrizes sparse;
- matrizes block-sparse;
- matrizes com padroes repetidos;
- matrizes reais extraidas de modelos pequenos.

### Perguntas

- Blocos iguais aparecem naturalmente?
- Blocos parecidos aparecem apos quantizacao?
- O codebook aprende padroes reutilizaveis?
- Multi-escala melhora em relacao a `2x2` puro?
- O custo do agrupamento compensa?

### Criterio de conclusao

Prosseguir somente se o codebook multi-escala mostrar vantagem clara em pelo menos um destes pontos:

- menor erro com mesmo numero de parametros;
- menos parametros com erro parecido;
- alta taxa de reutilizacao;
- reconstrucao eficiente o suficiente para uso em treino.

## 4. Fase 3 - Roteador de Blocos

Status: **pendente**.

### Objetivo

Criar o mecanismo que decide como cada regiao da matriz deve ser representada.

### Politicas Iniciais

- congelar;
- bloco grande;
- bloco medio;
- bloco pequeno;
- mistura multi-escala;
- delta livre;
- LoRA auxiliar.

### Heuristicas

```text
erro baixo + baixa sensibilidade  -> congelar ou bloco grande
erro alto + baixa sensibilidade   -> bloco medio
erro baixo + alta sensibilidade   -> bloco pequeno
erro alto + alta sensibilidade    -> bloco pequeno + LoRA/delta livre
```

### Entradas do Roteador

- erro de reconstrucao;
- norma do bloco;
- assinatura;
- sensibilidade;
- ganho por byte;
- tipo de matriz;
- camada;
- limite de VRAM.

### Criterio de conclusao

O roteador deve produzir um plano explicavel:

```text
regiao A -> congelada
regiao B -> codebook 8x8
regiao C -> codebook 4x4 + refinamento 2x2
regiao D -> delta livre
```

## 5. Fase 4 - Treino de Camada Linear

Status: **pendente**.

### Objetivo

Testar aprendizado antes de usar Transformer.

### Tarefa

Treinar uma camada linear para aproximar uma funcao alvo.

Comparar:

- treino full da matriz;
- LoRA;
- bloco `2x2`;
- codebook `2x2`;
- codebook multi-escala;
- codebook multi-escala com roteador;
- mascara esparsa por sensibilidade.

### Medidas

- loss final;
- velocidade de convergencia;
- parametros treinaveis;
- memoria do otimizador;
- erro de reconstrucao;
- estabilidade;
- capacidade de generalizacao.

### Criterio de conclusao

SAINT deve empatar ou superar pelo menos uma baseline eficiente em algum eixo relevante:

- menos memoria;
- menos parametros;
- menor checkpoint;
- melhor loss para mesmo orcamento;
- melhor ganho por byte.

## 6. Fase 5 - Mini-Transformer

Status: **pendente**.

### Objetivo

Validar SAINT em um modelo com acoplamento real entre camadas.

### Modelo

Um transformer pequeno, por exemplo:

```text
vocab_size: pequeno
n_layers: 2 a 6
d_model: 128 a 512
n_heads: 4 a 8
seq_len: 128 a 512
```

### Modos

- treino tradicional;
- head-only;
- LoRA;
- SAINT por camada;
- SAINT por matriz;
- SAINT por blocos;
- SAINT com codebook multi-escala;
- SAINT com consolidacao.

### Perguntas

- Loss global com atualizacao local funciona?
- Treino por partes converge?
- Consolidacao reduz conflito?
- A ordem de treino importa muito?
- O mapa de sensibilidade escolhe boas partes?

### Criterio de conclusao

Prosseguir se SAINT aprender de forma consistente e nao apenas memorizar comportamento acidental.

## 7. Fase 6 - Mapa de Sensibilidade

Status: **pendente**.

### Objetivo

Escolher os blocos certos para treinar.

### Metodos

- norma do gradiente;
- gradiente vezes peso;
- impacto de mascaramento;
- estimativa Fisher;
- magnitude de ativacao;
- erro por camada;
- ganho por byte.

### Experimentos

Comparar selecao:

- aleatoria;
- por camada fixa;
- por norma;
- por impacto na loss;
- por erro de reconstrucao;
- por ganho por byte;
- por frequencia de padrao.

### Criterio de conclusao

O mapa de sensibilidade deve superar selecao aleatoria de forma clara.

## 8. Fase 7 - Runtime SAINT

Status: **pendente**.

### Objetivo

Criar o runtime unificado.

### Modulos

```text
saint/
  config/
  blocks/
  codebook/
  routing/
  sensitivity/
  memory/
  training/
  checkpoints/
  adapters/
  cli/
```

### Funcionalidades

- CLI `saint`;
- config YAML/JSON;
- memory planner;
- particionador de pesos;
- roteador de blocos;
- codebook manager;
- mapa de sensibilidade;
- trainer;
- checkpoint manager;
- logger;
- resume.

### Comandos

```bash
saint inspect --model ./model
saint reconstruct --matrix ./weights.pt
saint estimate --model ./model --vram-gb 12
saint train --config configs/exp.yaml
saint resume --run runs/exp001
saint merge --run runs/exp001
```

### Criterio de conclusao

O runtime deve executar experimentos pequenos de ponta a ponta com logs e checkpoints.

## 9. Fase 8 - Checkpoint e Reconstituicao

Status: **pendente**.

### Objetivo

Salvar deltas parciais e recompor o modelo final.

### Formatos

- checkpoint composto;
- checkpoint fundido;
- apenas deltas;
- apenas codebook;
- roteador + codebook + escalas;
- estado do otimizador da parte ativa;
- historico de scheduler.

### Estrutura

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
    merged/
```

### Criterio de conclusao

SAINT deve conseguir:

```text
treinar -> salvar -> retomar -> fundir -> avaliar
```

## 10. Fase 9 - Adaptador DRM Transformer

Status: **pendente**.

### Objetivo

Usar `drm_transformer` como primeiro modelo customizado.

### Entregas

- adapter para carregar `DRMTransformer`;
- listagem de camadas e matrizes;
- congelar/descongelar partes;
- aplicar delta SAINT;
- salvar deltas;
- avaliar loss;
- comparar contra treino tradicional pequeno.

### Criterio de conclusao

O `drm_transformer` deve treinar em modo SAINT em escala pequena e produzir logs comparaveis.

## 11. Fase 10 - Modelos Hugging Face Pequenos

Status: **pendente**.

### Objetivo

Testar SAINT em modelos reais pequenos.

### Modelos Alvo

- modelos abaixo de 1B;
- depois 1B a 3B;
- modelos causal LM simples;
- modelos com arquitetura aberta.

### Experimentos

- fine-tuning pequeno;
- comparacao contra LoRA;
- comparacao contra QLoRA quando aplicavel;
- avaliacao de perplexity;
- avaliacao qualitativa de geracao.

### Criterio de conclusao

SAINT deve mostrar vantagem ou comportamento complementar a LoRA em pelo menos um tipo de tarefa.

## 12. Fase 11 - Escala 3B

Status: **pendente**.

### Objetivo

Primeiro teste serio em GPU domestica.

### Configuracao Esperada

```text
modelo: 3B
VRAM alvo: 12GB ou 24GB
modo: base congelada + deltas SAINT
seq_len: 512 a 2048
micro_batch: 1
offload: opcional
```

### Criterios

- rodar sem OOM;
- salvar checkpoints pequenos;
- melhorar loss;
- comparar com LoRA;
- medir tokens/s;
- medir ganho por byte.

## 13. Fase 12 - Escala 14B

Status: **pendente**.

### Objetivo

Testar gargalos de offload e roteamento.

### Foco

- memory planner;
- offload CPU;
- baixa VRAM;
- codebook multi-escala;
- consolidacao;
- scheduler por sensibilidade.

### Riscos

- lentidao excessiva;
- overhead de roteamento;
- baixa taxa de reutilizacao;
- conflito entre partes;
- ganho menor que LoRA.

### Criterio de conclusao

Prosseguir para 70B somente se 14B demonstrar:

- treino estavel;
- memoria controlada;
- ganho mensuravel;
- tempo aceitavel;
- alguma vantagem contra baseline.

## 14. Fase 13 - Escala 70B

Status: **pendente**.

### Objetivo

Validar SAINT como adaptacao extrema em hardware limitado.

### Condicoes

Nao e treino full de 70B.

E:

```text
modelo base quantizado/congelado
deltas SAINT esparsos
codebook multi-escala
offload agressivo
micro-batch 1
dataset pequeno
loss global
atualizacao local
```

### Experimentos

- estimativa de memoria;
- inferencia com deltas;
- fine-tuning pequeno;
- comparacao contra QLoRA quando possivel;
- medicao de tempo real;
- analise de qualidade.

### Criterio de sucesso

O objetivo minimo:

```text
rodar um ciclo completo de treino parcial em 70B
sem OOM
com checkpoint recomponivel
e alguma melhoria mensuravel na loss
```

## 15. Fase 14 - Otimizacoes

Status: **pendente**.

### Objetivo

Reduzir overhead.

### Possibilidades

- kernels customizados;
- agrupamento por codebook id;
- cache de blocos;
- sparse operations;
- quantizacao dos deltas;
- optimizer states em CPU;
- prefetch de blocos;
- compressao de checkpoints;
- compilacao de planos estaticos;
- batch de operacoes por grupo.

### Criterio de conclusao

O overhead do SAINT deve ficar pequeno o bastante para ser pratico em experimentos reais.

## 16. Fase 15 - Avaliacao

Status: **pendente**.

### Objetivo

Medir qualidade alem da loss.

### Avaliacoes

- perplexity;
- tarefas simples de QA;
- instrucao-resposta;
- codigo pequeno;
- retencao do modelo base;
- degradacao apos merge;
- comparacao contra LoRA/QLoRA;
- robustez em datasets diferentes.

### Perguntas

- SAINT aprende algo util?
- SAINT destroi capacidades antigas?
- SAINT e mais eficiente que LoRA em algum regime?
- SAINT escala com modelo maior?
- SAINT depende demais do dataset?

## 17. Fase 16 - Produto de Pesquisa

Status: **pendente**.

### Objetivo

Transformar o projeto em ferramenta utilizavel.

### Entregas

- README completo;
- exemplos;
- configs prontas;
- resultados reproduziveis;
- scripts de benchmark;
- notebooks opcionais;
- documentacao tecnica;
- guia de experimentos;
- modelo pequeno demonstrativo.

### Comandos Esperados

```bash
saint estimate --model tiny
saint reconstruct --checkpoint runs/matrix001
saint train --config configs/tiny_transformer.yaml
saint compare --run runs/saint001 --baseline runs/lora001
saint merge --run runs/saint001 --out merged/
```

## 18. Ordem Recomendada

Prioridade pratica:

```text
0. fundacao conceitual concluida
1. biblioteca de blocos
2. benchmark de reconstrucao
3. codebook multi-escala
4. roteador
5. camada linear treinavel
6. mini-transformer
7. mapa de sensibilidade
8. runtime CLI
9. drm_transformer
10. modelos HF pequenos
11. 3B
12. 14B
13. 70B
```

## 18.1 Proxima Acao Imediata

Criar a base de implementacao da Fase 1:

```text
saint/
  __init__.py
  blocks/
    __init__.py
    partition.py
    signatures.py
    grouping.py
tests/
  test_blocks_partition.py
  test_blocks_signatures.py
```

Primeiro objetivo tecnico:

```text
particionar matriz -> reconstruir matriz identica
```

Segundo objetivo tecnico:

```text
detectar blocos iguais por assinatura simples
```

## 19. Sinais de Alerta

Parar e reavaliar se:

- codebook nao comprime matrizes melhor que baselines simples;
- agrupamento de blocos quase nunca reutiliza padroes;
- roteador escolhe regioes sem ganho real;
- SAINT perde sempre para LoRA em memoria e qualidade;
- offload torna treino impraticavel;
- consolidacao nao reduz conflito;
- mini-transformer nao converge.

## 20. Marcos

### Marco A - Prova de Matriz

```text
SAINT representa matrizes com codebook multi-escala
e mostra compressao/erro competitivo.
```

### Marco B - Prova de Aprendizado

```text
SAINT treina uma camada linear melhor que uma baseline simples
em pelo menos um regime de memoria.
```

### Marco C - Prova de Acoplamento

```text
SAINT treina um mini-transformer com loss global e atualizacao local.
```

### Marco D - Prova de Modelo Real

```text
SAINT adapta um modelo real pequeno e compara com LoRA.
```

### Marco E - Prova de Escala

```text
SAINT roda em 3B/14B com memoria controlada.
```

### Marco F - Prova Extrema

```text
SAINT executa treino parcial em 70B com checkpoint recomponivel.
```

## 21. Definicao de Sucesso

SAINT sera considerado promissor se demonstrar pelo menos uma destas vantagens:

- menor memoria que LoRA em algum regime;
- menor checkpoint;
- melhor ganho por parametro treinavel;
- melhor ganho por byte de VRAM;
- boa adaptacao com base congelada;
- recomposicao estavel;
- codebook reutilizavel entre camadas ou modelos.

SAINT sera considerado fraco se:

- sempre perder para LoRA/QLoRA;
- exigir overhead alto demais;
- nao convergir em modelos pequenos;
- depender de ajustes manuais excessivos;
- nao reutilizar padroes de forma significativa.

## 22. Resumo

O desenvolvimento deve seguir uma regra:

```text
nao escalar antes de provar
```

Cada fase precisa produzir uma resposta objetiva antes da proxima.

O caminho seguro e:

```text
provar representacao
provar aprendizado
provar acoplamento
provar runtime
provar escala
```
