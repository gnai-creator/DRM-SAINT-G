# Glossario Tecnico SAINT

Este glossario define os termos usados no projeto SAINT.

SAINT significa **Simple AI Node Training**.

Definicao curta:

```text
SAINT = sparse multi-scale block-codebook delta training
```

Em portugues:

```text
treino de deltas esparsos por dicionario multi-escala de blocos
```

## Adapter

Camada de integracao entre o runtime SAINT e uma arquitetura de modelo especifica.

Responsabilidades:

- carregar o modelo;
- listar camadas e matrizes;
- congelar parametros;
- anexar deltas SAINT;
- executar forward;
- salvar e carregar deltas.

Exemplos:

- `DRMTransformerAdapter`;
- `HFCausalLMAdapter`;
- `MiniTransformerAdapter`.

## Arquitetura Decoder-Only

Arquitetura de Transformer usada por muitas LLMs autoregressivas.

O modelo recebe tokens anteriores e prediz o proximo token.

## Assinatura de Bloco

Representacao compacta usada para identificar ou agrupar blocos de matriz.

Pode usar:

- valores quantizados;
- norma;
- traco;
- determinante;
- autovalores;
- hash aproximado;
- cluster;
- id de codebook.

A assinatura nao precisa reconstruir o bloco sozinha. Ela serve para comparar e agrupar.

## Ativacoes

Valores intermediarios produzidos durante o forward.

Sao necessarios para o backward tradicional e podem consumir muita memoria.

## Backward

Etapa que calcula gradientes a partir da loss.

No SAINT, o forward pode envolver o modelo inteiro, mas o backward deve atualizar apenas as partes ativas.

## Baseline

Metodo de comparacao.

SAINT deve ser comparado contra:

- treino completo em modelos pequenos;
- LoRA;
- QLoRA;
- adapters;
- head-only;
- blocos sem codebook;
- codebook de tamanho unico;
- codebook multi-escala.

## Bloco

Submatriz pequena extraida de uma matriz maior.

Exemplos:

- `2x2`;
- `3x3`;
- `4x4`;
- `8x8`;
- `16x16`.

Uma matriz `4096x4096` pode ser vista como uma grade de blocos menores.

## Bloco Livre

Bloco treinavel sem restricao de codebook.

Tem maior expressividade, mas custa mais parametros e memoria.

Deve ser usado apenas em regioes criticas.

## Block Router

Modulo que decide como cada regiao da matriz sera representada.

Possibilidades:

- congelar;
- bloco grande;
- bloco pequeno;
- mistura multi-escala;
- codebook;
- delta livre;
- LoRA auxiliar.

## Cache de Grupos

Cache usado quando muitos blocos compartilham o mesmo prototipo.

Ideia:

```text
calcular prototipo uma vez
aplicar em varias posicoes
acumular gradientes no mesmo prototipo
```

## Checkpoint Composto

Checkpoint que salva o modelo base separado dos deltas SAINT.

Vantagem:

- menor tamanho;
- reversivel;
- permite trocar deltas.

## Checkpoint Fundido

Checkpoint onde os deltas foram materializados nos pesos finais.

Forma:

```text
W_final = W_base + DeltaW
```

## Codebook

Dicionario de blocos prototipo.

Em vez de treinar cada bloco separadamente, SAINT pode treinar prototipos reutilizaveis.

Exemplo:

```text
Dij = escala_ij * codebook[id_ij]
```

## Codebook Multi-Escala

Conjunto de codebooks com varios tamanhos de bloco.

Exemplo:

```text
codebook_2x2
codebook_4x4
codebook_8x8
codebook_16x16
```

Permite representar regioes simples com blocos maiores e regioes complexas com blocos menores.

## Consolidacao

Fase de revisao apos treinar partes separadas.

Objetivos:

- reduzir conflito entre deltas;
- melhorar consistencia global;
- recalcular sensibilidade;
- estabilizar a recomposicao.

## Delta

Atualizacao aplicada sobre uma matriz base.

Forma:

```text
W_eff = W_base + DeltaW
```

No SAINT, `DeltaW` e o principal objeto treinavel.

## Delta Esparso

Delta que altera apenas uma parte pequena da matriz.

Beneficios:

- menos parametros treinaveis;
- menos gradientes;
- menor optimizer state;
- checkpoints menores.

## Embedding

Tabela que transforma token IDs em vetores.

Shape comum:

```text
vocab_size x d_model
```

## Erro de Reconstrucao

Diferenca entre matriz original e matriz aproximada.

Exemplo:

```text
erro = ||W - W_aprox||
```

Usado para avaliar codebooks antes de treinar uma LLM.

## Fisher

Estimativa de importancia de parametros baseada na curvatura ou sensibilidade da loss.

Pode ser usada para escolher partes importantes.

## Forward

Etapa em que o modelo calcula logits e loss.

No SAINT, o forward deve refletir o modelo inteiro com deltas ativos.

## Ganho por Byte

Metrica de eficiencia.

Forma conceitual:

```text
ganho_estimado_de_loss / bytes_de_memoria_treinavel
```

Usada para priorizar partes que justificam o custo de memoria.

## Gradiente Local

Gradiente calculado apenas para a parte ativa.

No SAINT:

```text
loss global
gradiente local
```

## Head-Only

Modo de treino onde apenas a cabeca final e treinada.

Serve como baseline simples.

## LoRA

Metodo de fine-tuning eficiente que aprende uma atualizacao de baixo rank.

Forma:

```text
DeltaW = A B
```

SAINT deve ser comparado contra LoRA.

## Loss Global

Loss calculada pelo modelo inteiro.

Mesmo que apenas alguns blocos sejam treinados, a loss deve medir o comportamento global.

## Mascara Esparsa

Mascara que define quais blocos ou regioes estao ativos.

Exemplo:

```text
Mij = 1 -> bloco treinavel
Mij = 0 -> bloco congelado
```

## Matrix Inspector

Modulo que localiza e classifica matrizes dentro do modelo.

Exemplos:

- `attention.Wq`;
- `attention.Wk`;
- `attention.Wv`;
- `mlp.up`;
- `mlp.down`;
- `lm_head`.

## Memory Planner

Modulo que estima e controla uso de memoria.

Considera:

- pesos base;
- deltas;
- gradientes;
- optimizer states;
- ativacoes;
- caches;
- offload;
- margem de seguranca.

## Merge

Processo de materializar deltas nos pesos finais.

Forma:

```text
W_final = W_base + DeltaW
```

## Micro-Batch

Batch pequeno processado por vez para caber na memoria.

Usado com gradient accumulation.

## Multi-Escala

Representacao que usa blocos de tamanhos diferentes.

Exemplo:

```text
16x16 para regioes simples
4x4 para regioes medias
2x2 para regioes complexas
```

## Offload

Movimento de pesos, estados ou buffers para CPU/RAM/NVMe.

Reduz VRAM, mas aumenta tempo.

## Optimizer State

Estado interno do otimizador.

AdamW, por exemplo, guarda momentos `m` e `v`.

Esse estado pode consumir muita memoria no treino tradicional.

## Orcamento por Camada

Distribuicao do limite de treino entre camadas e matrizes.

Exemplo:

```text
camada 10: mais blocos treinaveis
camada 2: menos blocos treinaveis
mlp.up: mais orcamento
attention.Wk: menos orcamento
```

## QLoRA

Metodo que combina modelo base quantizado com LoRA treinavel.

E baseline importante para modelos grandes.

## Reconstrucao

Processo de gerar uma matriz aproximada a partir de blocos, codebooks e escalas.

Forma:

```text
codebooks + ids + escalas -> DeltaW ou W_aprox
```

## Recomposer

Modulo que reconstrui deltas ou funde deltas ao modelo.

## Roteador Heuristico

Roteador baseado em regras, nao treinavel.

Exemplo:

```text
erro alto + alta sensibilidade -> bloco pequeno ou delta livre
erro baixo + baixa sensibilidade -> congelar ou bloco grande
```

## Scheduler de Fases

Modulo que decide a ordem do treino.

Pode usar:

- sensibilidade;
- ganho por byte;
- erro de reconstrucao;
- reuso de padrao;
- camada;
- tipo de matriz.

## Sensibilidade

Medida de importancia de uma parte do modelo.

Pode ser medida por:

- norma do gradiente;
- impacto na loss;
- magnitude de ativacao;
- estimativa Fisher;
- erro de reconstrucao.

## Sparse Multi-Scale Block-Codebook Delta Training

Nome tecnico do paradigma SAINT.

Significa:

- `sparse`: nem tudo e treinado;
- `multi-scale`: usa blocos de tamanhos diferentes;
- `block-codebook`: usa dicionario de blocos;
- `delta training`: treina atualizacoes sobre pesos base.

## Token

Unidade numerica produzida pelo tokenizer.

LLMs treinam sequencias de tokens.

## Treino Parcial

Treino onde apenas uma parte dos parametros recebe gradiente.

No SAINT, o treino parcial deve ser guiado por loss global.

## VRAM

Memoria da GPU.

SAINT usa VRAM como restricao central do plano de treino.

