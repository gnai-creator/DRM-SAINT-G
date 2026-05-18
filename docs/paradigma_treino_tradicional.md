# Paradigma Tradicional de Treino de LLMs

Este documento descreve como funciona o paradigma atual de treinamento de uma LLM tradicional. O objetivo e explicar o modelo mental dominante: como os dados entram, como o modelo calcula previsoes, como a loss e calculada, como os pesos sao atualizados e por que o custo de memoria e computacao cresce tanto.

## 1. Ideia Central

Uma LLM tradicional e treinada para prever o proximo token.

Dado um trecho de texto:

```text
O gato subiu no telhado
```

Depois da tokenizacao, o texto vira uma sequencia de IDs:

```text
[O, gato, subiu, no, telhado]
```

O treino transforma isso em pares entrada-alvo:

```text
entrada: [O, gato, subiu, no]
alvo:    [gato, subiu, no, telhado]
```

O modelo recebe a entrada e tenta prever, em cada posicao, qual e o proximo token correto.

Esse tipo de treino e chamado de **autoregressive language modeling** ou **causal language modeling**.

## 2. O Que o Modelo Aprende

O modelo nao aprende frases diretamente. Ele aprende os pesos de uma rede neural que aproximam uma distribuicao de probabilidade:

```text
P(proximo_token | tokens_anteriores)
```

Exemplo:

```text
P("telhado" | "O gato subiu no")
```

Durante o treino, a rede ajusta bilhoes de numeros internos para melhorar essa previsao em muitos contextos diferentes.

Em uma LLM moderna, esses numeros internos incluem:

- embeddings de tokens;
- pesos de attention;
- pesos de feed-forward networks;
- normalizacoes;
- bias, quando usados;
- cabeca final de predicao do vocabulario.

## 3. Pipeline Geral

O paradigma tradicional pode ser visto assim:

```text
texto bruto
  -> limpeza e filtragem
  -> tokenizacao
  -> sequencias de tokens
  -> batches
  -> forward pass
  -> loss
  -> backward pass
  -> optimizer step
  -> checkpoint
```

Cada etapa tem impacto direto em custo, qualidade e estabilidade.

## 4. Dados

### 4.1 Coleta

LLMs sao treinadas com grandes quantidades de texto.

Fontes comuns:

- paginas web;
- livros;
- artigos;
- codigo;
- documentacao;
- conversas;
- dados sinteticos;
- datasets academicos;
- dados proprietarios.

### 4.2 Limpeza

O texto bruto normalmente contem:

- duplicatas;
- spam;
- boilerplate;
- HTML;
- conteudo quebrado;
- textos de baixa qualidade;
- dados sensiveis;
- exemplos contaminados de benchmarks.

Antes do treino, aplica-se uma etapa de filtragem.

Objetivos:

- reduzir ruido;
- remover duplicatas;
- melhorar qualidade linguistica;
- reduzir contaminacao;
- balancear dominios;
- remover conteudo proibido ou indesejado.

### 4.3 Tokenizacao

O texto nao entra no modelo como caracteres ou palavras puras. Ele e convertido em tokens.

Um tokenizer divide texto em unidades numericas.

Exemplo simplificado:

```text
"treinamento de IA"
```

Pode virar:

```text
[1532, 87, 421]
```

Cada numero e um ID no vocabulario.

Tokenizadores comuns usam variantes de:

- BPE;
- WordPiece;
- Unigram;
- SentencePiece.

O vocabulario costuma ter dezenas ou centenas de milhares de tokens.

## 5. Sequencias

Depois da tokenizacao, os tokens sao agrupados em sequencias de tamanho fixo ou maximo.

Exemplo com `seq_len = 8`:

```text
tokens:
[10, 25, 31, 44, 90, 13, 57, 61, 72]

input:
[10, 25, 31, 44, 90, 13, 57, 61]

target:
[25, 31, 44, 90, 13, 57, 61, 72]
```

O modelo calcula uma previsao para cada posicao.

Em treino causal, a posicao 3 pode usar apenas tokens ate a posicao 3. Ela nao pode olhar o futuro.

Isso e imposto por uma **causal mask**.

## 6. Arquitetura Transformer Decoder-Only

A maioria das LLMs modernas usa uma arquitetura Transformer decoder-only.

Estrutura geral:

```text
tokens
  -> token embeddings
  -> positional information
  -> bloco transformer 1
  -> bloco transformer 2
  -> ...
  -> bloco transformer N
  -> final normalization
  -> linear head
  -> logits sobre vocabulario
```

### 6.1 Embeddings

Cada token ID e convertido em um vetor.

Se `d_model = 4096`, cada token vira um vetor com 4096 numeros.

```text
token_id -> embedding[token_id] -> vetor
```

Esses embeddings sao parametros treinaveis.

### 6.2 Posicao

Como attention por si so nao sabe a ordem dos tokens, o modelo precisa de informacao posicional.

Formas comuns:

- positional embeddings aprendidos;
- sinusoidal embeddings;
- RoPE;
- ALiBi;
- variantes de positional bias.

Em LLMs modernas, RoPE e muito comum.

### 6.3 Bloco Transformer

Um bloco tipico tem:

```text
x
  -> norm
  -> self-attention causal
  -> residual
  -> norm
  -> feed-forward / MLP
  -> residual
```

Em forma simplificada:

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

Esse padrao e chamado de **pre-norm transformer**.

## 7. Self-Attention

Self-attention e o mecanismo que permite que cada token combine informacao dos tokens anteriores.

Para cada token, o modelo cria tres vetores:

- Query `Q`;
- Key `K`;
- Value `V`.

Eles sao produzidos por multiplicacoes lineares:

```text
Q = X Wq
K = X Wk
V = X Wv
```

Depois calcula-se similaridade entre queries e keys:

```text
scores = Q K^T / sqrt(d_head)
```

Aplica-se a causal mask para impedir olhar o futuro.

Depois:

```text
weights = softmax(scores)
output = weights V
```

O resultado passa por outra projecao linear.

### 7.1 Multi-Head Attention

Em vez de fazer uma attention unica, o modelo divide o espaco em varias heads.

Cada head aprende padroes diferentes.

Exemplos intuitivos:

- uma head pode rastrear concordancia gramatical;
- outra pode rastrear nomes;
- outra pode focar em delimitadores de codigo;
- outra pode operar sobre estrutura de listas.

Na pratica, nao ha garantia interpretavel simples para cada head, mas essa e a intuicao.

### 7.2 Custo da Attention

Attention tradicional tem custo quadratico no comprimento da sequencia:

```text
O(seq_len^2)
```

Se dobramos o contexto, o custo da matriz de attention cresce aproximadamente quatro vezes.

Por isso contexto longo e caro.

## 8. MLP / Feed-Forward

Depois da attention, cada posicao passa por uma rede feed-forward.

Forma comum:

```text
MLP(x) = W2 activation(W1 x)
```

Muitas LLMs usam variantes como:

- GELU;
- SiLU;
- SwiGLU;
- GeGLU.

Em muitos modelos, a MLP concentra grande parte dos parametros.

Exemplo:

```text
d_model = 4096
d_ff = 11008
```

As matrizes da MLP ficam muito grandes.

## 9. Logits e Predicao

No final, o modelo gera logits.

Se o vocabulario tem 50.000 tokens, para cada posicao da sequencia o modelo gera 50.000 numeros.

```text
hidden_state -> linear -> logits[vocab_size]
```

Esses logits sao convertidos em probabilidades pelo softmax.

Durante o treino, eles sao comparados com o token correto.

## 10. Loss

A loss padrao e **cross entropy**.

Para cada posicao, o modelo atribui uma probabilidade ao token correto.

Se a probabilidade for alta, loss baixa.

Se a probabilidade for baixa, loss alta.

Forma conceitual:

```text
loss = -log P(token_correto)
```

Para uma sequencia, calcula-se a media sobre todos os tokens.

Para um batch, calcula-se a media sobre todas as sequencias.

## 11. Forward Pass

O forward pass e a etapa em que o modelo calcula a saida.

Entrada:

```text
batch de tokens [batch_size, seq_len]
```

Saida:

```text
logits [batch_size, seq_len, vocab_size]
loss escalar
```

Durante o forward, o framework salva ativacoes intermediarias que serao usadas no backward.

Essas ativacoes ocupam muita memoria.

## 12. Backward Pass

O backward pass calcula gradientes.

Gradiente significa:

```text
quanto cada peso contribuiu para o erro
```

Para cada parametro treinavel, o sistema calcula:

```text
d loss / d parametro
```

Isso e feito por backpropagation.

O PyTorch, por exemplo, constroi um grafo automatico durante o forward e usa esse grafo para calcular os gradientes no backward.

## 13. Optimizer Step

Depois que os gradientes sao calculados, o otimizador atualiza os pesos.

O otimizador mais comum em LLMs e AdamW.

AdamW mantem estados adicionais por parametro:

- gradiente atual;
- media movel dos gradientes;
- media movel dos quadrados dos gradientes.

Isso melhora estabilidade, mas aumenta muito a memoria.

Para cada parametro, AdamW pode precisar de varios valores extras.

Em treino full precision tradicional, a memoria nao e apenas:

```text
parametros
```

E mais proximo de:

```text
parametros
+ gradientes
+ optimizer states
+ ativacoes
+ buffers temporarios
```

## 14. Batch, Micro-Batch e Gradient Accumulation

### 14.1 Batch

Batch e o conjunto de exemplos processados juntos.

Exemplo:

```text
batch_size = 8
seq_len = 2048
tokens por batch = 8 * 2048 = 16384
```

### 14.2 Micro-Batch

Quando a GPU nao comporta um batch grande, divide-se o batch em micro-batches.

Exemplo:

```text
batch efetivo desejado: 64
micro_batch_size: 1
gradient_accumulation_steps: 64
```

O modelo processa 64 micro-batches de tamanho 1, acumula gradientes e so depois atualiza os pesos.

### 14.3 Batch Efetivo

Formula:

```text
batch_efetivo = micro_batch_size * gradient_accumulation_steps * num_gpus
```

Em tokens:

```text
tokens_por_update = batch_efetivo * seq_len
```

Esse valor e importante para estabilidade do treino.

## 15. Mixed Precision

Treinar tudo em `fp32` e caro.

LLMs modernas geralmente usam:

- `fp16`;
- `bf16`;
- mistura de precisao baixa com acumulacao mais estavel.

### 15.1 FP32

Cada numero usa 4 bytes.

Mais estavel, mas consome muita memoria.

### 15.2 FP16

Cada numero usa 2 bytes.

Economiza memoria e e rapido em GPUs modernas, mas pode sofrer overflow/underflow.

### 15.3 BF16

Cada numero usa 2 bytes, mas com faixa numerica parecida com FP32.

Normalmente e mais estavel que FP16.

### 15.4 Loss Scaling

Com FP16, pode ser necessario usar loss scaling para evitar underflow dos gradientes.

## 16. Gradient Checkpointing

No paradigma tradicional, o forward salva muitas ativacoes para o backward.

Gradient checkpointing reduz memoria descartando algumas ativacoes e recalculando-as depois.

Troca:

```text
menos memoria
por
mais computacao
```

E muito usado em treino de modelos grandes.

## 17. Checkpoints

Durante o treino, salva-se o estado do processo.

Um checkpoint pode conter:

- pesos do modelo;
- estado do otimizador;
- estado do scheduler de learning rate;
- step atual;
- scaler de mixed precision;
- configuracao;
- estado do dataloader ou seed.

Checkpoint completo de modelo grande pode ocupar centenas de GB ou mais.

## 18. Learning Rate Schedule

O learning rate controla o tamanho das atualizacoes dos pesos.

LLMs normalmente usam:

1. warmup inicial;
2. decaimento gradual.

Exemplo:

```text
lr sobe de 0 ate lr_max nos primeiros N steps
depois cai com cosine decay ou linear decay
```

Warmup evita instabilidade no inicio.

## 19. Regularizacao

Algumas tecnicas:

- weight decay;
- dropout;
- data deduplication;
- early stopping em alguns contextos;
- gradient clipping;
- mixture/balanceamento de dados.

Em LLMs grandes, dropout pode ser baixo ou ate ausente dependendo do regime e escala.

## 20. Distribuicao em Multiplas GPUs

Modelos grandes normalmente precisam de varias GPUs.

Ha diferentes formas de paralelismo.

### 20.1 Data Parallelism

Cada GPU tem uma copia completa do modelo.

Cada GPU processa um batch diferente.

Depois os gradientes sao sincronizados.

Limite:

```text
o modelo inteiro precisa caber em cada GPU
```

### 20.2 Tensor Parallelism

Divide matrizes grandes entre GPUs.

Exemplo:

```text
uma camada linear gigante e partida em colunas ou linhas
```

Cada GPU guarda uma parte da camada.

### 20.3 Pipeline Parallelism

Divide camadas entre GPUs.

Exemplo:

```text
GPU 0: blocos 0-7
GPU 1: blocos 8-15
GPU 2: blocos 16-23
GPU 3: blocos 24-31
```

Micro-batches passam pelo pipeline.

### 20.4 ZeRO / Sharding

ZeRO divide estados do otimizador, gradientes e parametros entre GPUs.

Ideia:

```text
nenhuma GPU precisa guardar tudo sozinha
```

Estagios comuns:

- ZeRO-1: divide optimizer states;
- ZeRO-2: divide optimizer states e gradientes;
- ZeRO-3: divide optimizer states, gradientes e parametros.

FSDP segue uma ideia semelhante de sharding de parametros.

## 21. Por Que Memoria Cresce Tanto

Para um modelo com `N` parametros:

### 21.1 Apenas Pesos

Em fp16/bf16:

```text
memoria_pesos = N * 2 bytes
```

Exemplos aproximados:

```text
7B  -> 14 GB so pesos fp16
20B -> 40 GB so pesos fp16
70B -> 140 GB so pesos fp16
```

### 21.2 Gradientes

Gradientes tambem ocupam memoria.

Em fp16:

```text
gradientes ~= N * 2 bytes
```

### 21.3 AdamW

AdamW mantem dois momentos por parametro.

Se mantidos em fp32:

```text
m ~= N * 4 bytes
v ~= N * 4 bytes
```

### 21.4 Estimativa Simples

Treino tradicional com AdamW pode consumir, por parametro:

```text
peso fp16:      2 bytes
grad fp16:      2 bytes
master fp32:    4 bytes
adam m fp32:    4 bytes
adam v fp32:    4 bytes
total:         16 bytes por parametro
```

Isso sem contar ativacoes.

Entao:

```text
7B * 16 bytes  ~= 112 GB
70B * 16 bytes ~= 1120 GB
```

Esse e um dos motivos pelos quais treino completo de LLMs grandes exige clusters.

## 22. Ativacoes

Ativacoes sao valores intermediarios calculados no forward.

Elas dependem de:

- batch size;
- seq_len;
- d_model;
- numero de camadas;
- tipo de attention;
- precision;
- checkpointing.

Mesmo se os pesos couberem, as ativacoes podem causar OOM.

Contexto longo aumenta muito esse custo.

## 23. FLOPs e Tempo

Treinar LLMs nao e limitado apenas por memoria.

Tambem ha custo computacional.

Uma regra aproximada para treino autoregressivo e que o custo total escala com:

```text
parametros * tokens_treinados
```

Modelos grandes precisam de trilhoes de tokens e enormes quantidades de FLOPs.

Mesmo que fosse possivel fazer offload para caber em memoria, o tempo em uma GPU domestica poderia ser impraticavel.

## 24. Pretraining

Pretraining e o treino inicial em larga escala.

Caracteristicas:

- muitos dados;
- muitos tokens;
- todos ou quase todos os parametros treinaveis;
- objetivo de proximo token;
- custo muito alto;
- treinamento por dias, semanas ou meses em clusters.

O resultado e um modelo base.

Modelos base costumam ser bons em completar texto, mas nao necessariamente seguem instrucoes.

## 25. Fine-Tuning Supervisionado

Depois do pretraining, pode-se fazer supervised fine-tuning.

Dados no formato:

```text
instrucao -> resposta desejada
```

O objetivo ainda pode ser prever proximo token, mas agora sobre exemplos formatados como dialogo ou instrucao.

Isso ensina o modelo a responder de forma mais util.

## 26. Alignment

Depois do SFT, muitos modelos passam por alinhamento.

Tecnicas comuns:

- RLHF;
- RLAIF;
- DPO;
- IPO;
- ORPO;
- variantes de preference optimization.

Nessa fase, o objetivo nao e apenas prever texto, mas preferir respostas melhores segundo algum criterio humano, automatico ou misto.

## 27. Fine-Tuning Eficiente

Como treinar todos os pesos e caro, surgiram tecnicas PEFT.

PEFT significa **parameter-efficient fine-tuning**.

### 27.1 LoRA

LoRA congela os pesos originais e treina matrizes pequenas de baixo rank.

Em vez de atualizar uma matriz grande `W`, aprende-se uma atualizacao:

```text
W' = W + A B
```

Onde `A` e `B` sao pequenas comparadas a `W`.

Vantagem:

- muito menos parametros treinaveis;
- menor memoria;
- checkpoints pequenos;
- bom para GPUs domesticas.

### 27.2 Adapters

Adapters inserem pequenos modulos treinaveis entre camadas.

O modelo base fica congelado.

### 27.3 Prefix / Prompt Tuning

Treina vetores adicionais que condicionam o modelo sem alterar todos os pesos.

## 28. Quantizacao

Quantizacao reduz a precisao dos pesos.

Exemplos:

- int8;
- int4;
- nf4.

Isso reduz memoria, principalmente para inferencia e fine-tuning com base congelada.

Mas treinar todos os parametros diretamente em 4-bit nao e o paradigma tradicional de pretraining full.

Quantizacao e mais comum em:

- inferencia;
- QLoRA;
- fine-tuning eficiente.

## 29. QLoRA

QLoRA combina:

- modelo base quantizado em 4-bit;
- LoRA treinavel;
- otimizadores com menor memoria;
- offload quando necessario.

E um paradigma muito importante para treinar/adaptar modelos grandes em hardware limitado.

Mas ainda nao significa treinar todos os pesos do modelo base.

## 30. Paradigma Tradicional em Pseudocodigo

```python
model = Transformer(config)
optimizer = AdamW(model.parameters(), lr=lr)

for step, batch in enumerate(dataloader):
    input_ids = batch["input_ids"]
    targets = batch["targets"]

    logits = model(input_ids)
    loss = cross_entropy(logits, targets)

    loss.backward()

    if should_update(step):
        clip_grad_norm_(model.parameters())
        optimizer.step()
        optimizer.zero_grad()

    if should_save(step):
        save_checkpoint(model, optimizer, step)
```

Com mixed precision e accumulation:

```python
for step, batch in enumerate(dataloader):
    with autocast(dtype=bf16):
        logits = model(batch.input_ids)
        loss = cross_entropy(logits, batch.targets)
        loss = loss / accumulation_steps

    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 31. O Que E Atualizado

No treino completo tradicional:

```text
todos os parametros treinaveis recebem gradientes
todos os parametros sao atualizados pelo otimizador
```

Isso inclui:

- embeddings;
- attention;
- MLP;
- normalizacoes;
- lm head.

No fine-tuning eficiente:

```text
modelo base congelado
apenas pequenos modulos recebem gradientes
```

## 32. Por Que o Paradigma Funciona

O paradigma funciona porque:

1. prever proximo token cria um sinal supervisionado barato em qualquer texto;
2. a escala de dados cobre muitos padroes;
3. transformers conseguem modelar dependencias longas;
4. otimizadores modernos conseguem treinar redes profundas;
5. aumentar parametros e tokens geralmente melhora desempenho ate certos limites.

O ponto forte e simplicidade:

```text
texto -> tokens -> prever proximo token
```

O ponto fraco e custo:

```text
precisa de muitos dados, memoria, computacao e energia
```

## 33. Limites do Paradigma Atual

Principais limites:

- custo enorme de pretraining;
- dependencia de escala;
- dificuldade de atualizar conhecimento sem retreino/fine-tuning;
- contexto longo caro;
- pouca eficiencia amostral;
- risco de memorizar dados;
- interpretabilidade limitada;
- alinhamento separado do aprendizado base;
- necessidade de infraestrutura complexa;
- dificuldade de treinar modelos enormes em hardware pequeno.

## 34. Onde Um Novo Paradigma Poderia Atacar

Um novo paradigma de treino poderia tentar reduzir:

- necessidade de atualizar todos os parametros;
- custo de ativacoes;
- dependencia de backpropagation global;
- necessidade de manter optimizer states para tudo;
- custo quadratico da attention;
- acoplamento entre todos os blocos;
- quantidade de tokens necessaria;
- dependencia de clusters.

Possiveis direcoes:

- treino local por modulo;
- memoria externa;
- roteamento dinamico;
- compressao progressiva;
- modelos que crescem por blocos;
- aprendizado online controlado;
- world models com estados latentes reutilizaveis;
- destilacao continua;
- treinamento em espacos menores;
- atualizacoes esparsas;
- otimizadores sem estado global pesado.

## 35. Resumo

O paradigma tradicional de treino de LLMs e:

```text
1. coletar muito texto
2. limpar e tokenizar
3. montar sequencias
4. prever proximo token
5. calcular cross entropy
6. fazer backpropagation
7. atualizar todos os pesos com AdamW
8. repetir por muitos tokens
```

Ele e poderoso porque transforma texto bruto em supervisao quase ilimitada.

Ele e caro porque exige manter muitos parametros, gradientes, estados do otimizador e ativacoes ao mesmo tempo.

Para modelos pequenos, isso cabe em uma GPU comum. Para modelos grandes, o paradigma tradicional exige paralelismo, sharding, mixed precision, checkpointing e clusters com muitas GPUs.
