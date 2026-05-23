# Paradigma SAINT-G

SAINT-G significa **DRM por Enxerto com SAINT-G-Phi**. Neste documento, SAINT-G e descrito como um paradigma experimental de treinamento parcial para modelos grandes, pensado para maquinas com pouca VRAM em relacao ao tamanho total do modelo.

A ideia central nao e treinar todos os parametros de uma LLM gigante ao mesmo tempo. A ideia e **quebrar o treinamento em partes pequenas, treinar essas partes sob um orcamento fixo de memoria e depois recompor o modelo final**.

## 1. Motivacao

O paradigma tradicional de treino de LLMs exige que muitos componentes estejam ativos ao mesmo tempo:

- pesos;
- ativacoes;
- gradientes;
- estados do otimizador;
- buffers temporarios;
- batches;
- sequencias longas.

Para modelos grandes, isso torna o treino completo inviavel em GPUs domesticas.

SAINT-G nasce da seguinte pergunta:

```text
E se o modelo pudesse ser treinado em partes muito pequenas,
respeitando um limite fixo de VRAM,
e depois essas partes fossem juntadas em um modelo final?
```

## 2. Principio Central

O usuario define um orcamento de VRAM:

```bash
SAINT-G train --vram-gb 12 --model 70b
```

SAINT-G nao tenta colocar o treinamento completo dentro desse limite.

Em vez disso, ele cria um plano:

```text
modelo grande
  -> dividir em unidades treinaveis pequenas
  -> treinar uma unidade ou grupo pequeno por vez
  -> congelar o restante
  -> salvar atualizacoes parciais
  -> alternar para outras unidades
  -> revisar e consolidar
  -> recompor modelo final
```

## 3. Unidade Treinavel

Uma unidade treinavel pode ser:

- um bloco Transformer;
- uma matriz inteira;
- uma submatriz;
- um conjunto de blocos `2x2`;
- um adapter;
- uma atualizacao residual;
- uma pequena funcao geradora de pesos.

No SAINT-G, uma matriz grande pode ser vista como uma grade de blocos menores.

Exemplo:

```text
W 4x4 =
[a b c d]
[e f g h]
[i j k l]
[m n o p]
```

Dividida em blocos `2x2`:

```text
W =
[B11 B12]
[B21 B22]

B11 = [a b]
      [e f]

B12 = [c d]
      [g h]

B21 = [i j]
      [m n]

B22 = [k l]
      [o p]
```

Uma matriz `4x4` vira 4 blocos `2x2`.

Uma matriz `4096x4096` vira:

```text
2048 * 2048 = 4.194.304 blocos 2x2
```

## 4. Dividir Nao Basta

Apenas dividir uma matriz grande em blocos pequenos nao reduz memoria total.

Uma matriz `4096x4096` tem:

```text
16.777.216 numeros
```

Se for dividida em blocos `2x2`, ainda tera:

```text
4.194.304 blocos * 4 numeros = 16.777.216 numeros
```

Portanto, SAINT-G precisa fazer algo alem de dividir.

Possibilidades:

- treinar apenas alguns blocos por vez;
- congelar blocos nao ativos;
- parametrizar blocos com menos numeros;
- compartilhar blocos entre regioes;
- armazenar atualizacoes em forma compacta;
- aplicar blocos como delta sobre pesos base;
- usar offload para o restante do modelo.

Portanto, o SAINT-G nao deve ser entendido como:

```text
dividir tudo em blocos 2x2
treinar todos os blocos 2x2
juntar tudo no final
```

Isso provavelmente seria lento e pouco eficiente.

O caminho mais forte e:

```text
dividir em blocos
descobrir quais blocos importam
descobrir quais blocos sao iguais ou parecidos
treinar grupos compactos de blocos
reutilizar calculos entre blocos equivalentes
recompor o delta final
```

## 5. Representacao por Delta

Em vez de substituir a matriz original, SAINT-G pode aprender uma atualizacao:

```text
W_final = W_base + DeltaW
```

Onde:

- `W_base` e o peso original, possivelmente congelado e quantizado;
- `DeltaW` e uma atualizacao treinada em partes;
- `W_final` e a matriz efetiva usada no modelo.

O treinamento SAINT-G pode focar apenas em `DeltaW`.

Isso reduz o custo porque o modelo base nao precisa ter todos os parametros atualizados.

## 6. Delta em Blocos 2x2

`DeltaW` pode ser organizado como uma grade de blocos `2x2`:

```text
DeltaW =
[D11 D12 D13 ...]
[D21 D22 D23 ...]
[D31 D32 D33 ...]
```

Cada `Dij` e um bloco:

```text
[x y]
[z w]
```

Mas SAINT-G pode escolher diferentes formas de representar esse bloco.

### 6.1 Bloco Completo

Cada bloco tem 4 parametros:

```text
[a b]
[c d]
```

Vantagem:

- expressivo;
- simples.

Desvantagem:

- nao comprime parametros.

### 6.2 Bloco Diagonal

```text
[a 0]
[0 d]
```

Tem 2 parametros.

### 6.3 Bloco Escalar

```text
[s 0]
[0 s]
```

Tem 1 parametro.

### 6.4 Bloco Rotacional

```text
[cos t  -sin t]
[sin t   cos t]
```

Tem 1 parametro.

Pode representar uma rotacao local no espaco de features.

### 6.5 Bloco Escala + Rotacao

```text
s * [cos t  -sin t]
    [sin t   cos t]
```

Tem 2 parametros.

### 6.6 Bloco Gerado por Codigo

Um bloco pode ser escolhido de um dicionario:

```text
Dij = escala * codebook[id]
```

Onde:

- `codebook` contem blocos aprendidos ou predefinidos;
- `id` seleciona o bloco;
- `escala` ajusta intensidade.

Isso permite compartilhamento.

### 6.7 Blocos Iguais ou Parecidos

Uma melhoria importante e procurar blocos iguais ou semelhantes dentro das matrizes do modelo.

Exemplo:

```text
D12 ~= D98 ~= D301
```

Se varios blocos forem iguais, parecidos ou pertencerem ao mesmo padrao, SAINT-G pode representar todos por uma unica entrada compartilhada:

```text
D12  = escala_12  * codebook[k]
D98  = escala_98  * codebook[k]
D301 = escala_301 * codebook[k]
```

Assim, em vez de treinar tres blocos separados, SAINT-G treina:

- um bloco base compartilhado;
- pequenas escalas ou ajustes por posicao.

Isso cria um tipo de compressao estrutural.

### 6.8 Calculo Compartilhado de Blocos

Se muitos blocos usam o mesmo padrao, o calculo pode ser agrupado.

Em vez de executar:

```text
calcular bloco A na posicao 1
calcular bloco A na posicao 2
calcular bloco A na posicao 3
```

SAINT-G pode executar:

```text
calcular bloco A uma vez
aplicar resultado nas posicoes que usam A
```

Na pratica, isso exigiria um runtime que agrupe indices por `codebook_id` ou por assinatura de bloco.

Fluxo conceitual:

```text
matriz grande
  -> dividir em blocos 2x2
  -> gerar assinatura de cada bloco
  -> agrupar blocos iguais/parecidos
  -> calcular grupo uma vez
  -> espalhar resultado para as posicoes correspondentes
```

Isso pode reduzir custo quando ha repeticao real ou quando SAINT-G forca repeticao por parametrizacao.

### 6.9 Assinatura de Bloco

Para encontrar blocos iguais ou parecidos, cada bloco pode receber uma assinatura.

Assinaturas possiveis:

- valores quantizados do bloco;
- norma;
- traco;
- determinante;
- autovalores;
- direcao principal;
- hash aproximado;
- cluster aprendido;
- codigo de codebook.

Importante: a determinante sozinha nao reconstrui o bloco. Mas pode ser uma das estatisticas usadas para agrupar blocos.

Exemplo de assinatura simples:

```text
signature(Dij) = quantize([a, b, c, d])
```

Exemplo de assinatura estrutural:

```text
signature(Dij) = cluster(norm, trace, det, eigenvalues)
```

### 6.10 Codebook Treinavel

O codebook pode ser treinavel.

Nesse caso, SAINT-G nao treina cada bloco individualmente. Ele treina um conjunto pequeno de blocos prototipo:

```text
codebook[0], codebook[1], ..., codebook[K-1]
```

Cada posicao escolhe ou mistura esses prototipos:

```text
Dij = escala_ij * codebook[id_ij]
```

Ou:

```text
Dij = soma_k alpha_ijk * codebook[k]
```

Isso permite que milhoes de blocos sejam representados por poucos padroes compartilhados.

Essa e uma das ideias mais promissoras do SAINT-G:

```text
nao treinar todos os blocos;
treinar os padroes que muitos blocos reutilizam.
```

### 6.11 Codebook Multi-Escala

O codebook nao precisa ter apenas blocos `2x2`.

SAINT-G pode usar um dicionario escalavel de matrizes pequenas:

```text
codebook_2x2
codebook_3x3
codebook_4x4
codebook_5x5
codebook_6x6
...
```

Cada codebook guarda prototipos daquele tamanho.

Exemplo:

```text
codebook_2x2[17] = bloco 2x2
codebook_4x4[03] = bloco 4x4
codebook_8x8[91] = bloco 8x8
```

Assim, uma matriz grande pode ser representada por blocos de tamanhos diferentes:

```text
regioes simples       -> blocos maiores
regioes complexas     -> blocos menores
regioes muito sensiveis -> blocos livres ou LoRA auxiliar
```

Isso cria uma representacao adaptativa.

### 6.12 Escolha do Tamanho do Bloco

O tamanho ideal do bloco pode depender da regiao da matriz.

Blocos maiores:

- reduzem overhead;
- aumentam reutilizacao;
- representam padroes mais amplos;
- podem perder detalhe local.

Blocos menores:

- capturam detalhes finos;
- permitem atualizacao mais precisa;
- aumentam numero de blocos;
- podem gerar overhead alto.

SAINT-G pode escolher o tamanho por criterio de eficiencia:

```text
score = ganho_estimado_de_loss / custo_de_memoria
```

Ou:

```text
score = erro_de_reconstrucao / numero_de_parametros
```

Fluxo possivel:

```text
1. tenta representar uma regiao com bloco 8x8
2. se erro for baixo, mantem 8x8
3. se erro for alto, divide em 4x4
4. se ainda for alto, divide em 2x2
5. se for uma regiao critica, permite bloco livre ou delta especial
```

Esse processo e parecido com uma decomposicao hierarquica.

### 6.13 Arvore de Blocos

Uma matriz pode ser decomposta como uma arvore.

Exemplo:

```text
matriz W
  -> regioes 64x64
    -> algumas viram blocos 16x16
      -> algumas viram blocos 4x4
        -> algumas viram blocos 2x2
```

Regioes simples ficam em niveis altos.

Regioes complexas sao refinadas.

Isso evita aplicar `2x2` no modelo inteiro.

### 6.14 Dicionario Hierarquico

Os codebooks podem ser organizados de forma hierarquica:

```text
codebook_global
  -> codebook_por_camada
  -> codebook_por_tipo_de_matriz
  -> codebook_por_tamanho
```

Exemplo:

```text
attention.Wq.codebook_4x4
attention.Wv.codebook_2x2
mlp.up.codebook_8x8
mlp.down.codebook_4x4
```

Isso permite que SAINT-G aprenda padroes diferentes para:

- attention;
- MLP;
- embeddings;
- lm_head;
- camadas iniciais;
- camadas intermediarias;
- camadas finais.

### 6.15 Mistura de Codebooks

Uma regiao pode combinar varios prototipos.

Forma discreta:

```text
Dij = escala * codebook_4x4[id]
```

Forma continua:

```text
Dij = alpha1 * codebook_4x4[a]
    + alpha2 * codebook_4x4[b]
    + alpha3 * codebook_4x4[c]
```

Forma multi-escala:

```text
D_region = bloco_8x8
         + refinamento_4x4
         + refinamento_2x2
```

Isso permite um delta grosseiro mais barato, com refinamentos pequenos apenas onde necessario.

### 6.16 Separar Reconstrucao de Treino

SAINT-G tem dois problemas diferentes:

```text
A. representar ou reconstruir bem uma matriz grande
B. treinar essa representacao usando a loss global da LLM
```

Esses problemas nao devem ser confundidos.

Antes de testar SAINT-G em uma LLM grande, e preciso responder:

```text
O codebook multi-escala consegue representar matrizes grandes
com erro baixo e poucos parametros?
```

Depois vem a segunda pergunta:

```text
Essa representacao consegue aprender quando guiada pela loss global
do modelo inteiro?
```

Se a resposta para a primeira pergunta for ruim, a segunda provavelmente tambem sera.

Portanto, SAINT-G deve validar primeiro a compressao/reconstrucao de matrizes isoladas e so depois validar treino em modelos.

### 6.17 Roteador de Blocos

SAINT-G pode ter um roteador que decide como cada regiao da matriz sera representada.

Para cada regiao, o roteador escolhe:

```text
congelar
usar bloco 16x16
usar bloco 8x8
usar bloco 4x4
usar bloco 2x2
usar mistura multi-escala
usar LoRA auxiliar
usar delta livre
```

No inicio, esse roteador pode ser heuristico.

Exemplo de heuristica:

```text
erro baixo + baixa sensibilidade  -> congelar ou bloco grande
erro alto + baixa sensibilidade   -> bloco medio
erro baixo + alta sensibilidade   -> bloco pequeno
erro alto + alta sensibilidade    -> bloco pequeno + LoRA/delta livre
```

No futuro, o roteador pode ser treinavel.

Mas uma versao heuristica ja seria suficiente para testar o paradigma.

## 7. Treino Parcial

O SAINT-G treina apenas uma parte por vez.

Exemplo:

```text
fase 1: treinar blocos 2x2 da matriz Wq da camada 0
fase 2: treinar blocos 2x2 da matriz Wk da camada 0
fase 3: treinar blocos 2x2 da matriz Wv da camada 0
fase 4: treinar blocos 2x2 da MLP da camada 0
fase 5: repetir para camada 1
...
```

Durante uma fase:

```text
parte ativa: recebe gradiente
resto do modelo: congelado
```

O otimizador guarda estado apenas da parte ativa.

## 7.1 Mapa de Sensibilidade

Antes de treinar blocos, SAINT-G deve descobrir quais partes merecem treino.

Sem isso, o sistema pode gastar tempo atualizando regioes que quase nao afetam a loss.

O mapa de sensibilidade mede a importancia de:

- camadas;
- matrizes;
- submatrizes;
- blocos `2x2`;
- grupos de blocos;
- entradas do codebook.

Fluxo:

```text
1. rodar alguns batches pelo modelo
2. calcular loss global
3. medir sinais de importancia por parte
4. criar ranking de partes
5. treinar apenas as partes mais relevantes
6. recalcular o mapa periodicamente
```

Metricas possiveis:

- norma do gradiente;
- gradiente vezes peso;
- variacao de loss ao mascarar uma parte;
- magnitude da ativacao;
- estimativa Fisher;
- frequencia de uso;
- erro por camada;
- conflito com deltas anteriores.

O mapa de sensibilidade transforma SAINT-G de:

```text
treinar em pedacos
```

para:

```text
treinar os pedacos certos
```

## 7.2 Mascara Esparsa de Treino

SAINT-G deve evitar ativar todos os blocos.

Depois do mapa de sensibilidade, ele cria uma mascara:

```text
Mij = 1 se bloco Dij sera treinado
Mij = 0 se bloco Dij ficara congelado
```

O delta efetivo vira:

```text
DeltaW = M * reconstruct(blocos)
```

Onde `M` e uma mascara esparsa.

Vantagens:

- menos parametros treinaveis;
- menos gradientes;
- menos optimizer state;
- checkpoints menores;
- menos risco de alterar partes irrelevantes.

Essa mascara pode ser:

- fixa por fase;
- atualizada a cada ciclo;
- baseada em threshold;
- limitada por orcamento de VRAM;
- limitada por numero maximo de blocos ativos.

## 7.3 Refinamento Coarse-to-Fine

Comecar diretamente em blocos `2x2` pode ser caro demais.

Uma estrategia melhor e coarse-to-fine:

```text
fase 1: medir camadas importantes
fase 2: medir matrizes importantes
fase 3: medir submatrizes 64x64 ou 32x32
fase 4: refinar para 16x16
fase 5: refinar para 2x2 apenas onde vale a pena
```

Assim, SAINT-G nao procura detalhes finos no modelo inteiro.

Ele primeiro localiza regioes promissoras e so depois aplica blocos pequenos.

## 8. Loss Global, Atualizacao Local

Este e um ponto essencial.

Mesmo que SAINT-G treine uma parte pequena, a qualidade dessa parte deve ser medida pela loss do modelo completo.

Fluxo:

```text
entrada
  -> modelo inteiro
  -> loss final
  -> gradiente apenas na parte ativa
  -> atualiza parte ativa
```

Ou seja:

```text
loss global
atualizacao local
```

Se cada parte for treinada isoladamente sem passar pelo modelo completo, ela pode nao funcionar quando todas forem juntadas.

## 9. Problema de Dependencia Entre Partes

As partes de uma LLM sao acopladas.

Um bloco muda a distribuicao de ativacoes que chega ao proximo bloco.

Se treinarmos partes separadas sem revisao, pode acontecer:

- uma parte desfaz o ganho da anterior;
- uma camada se adapta a outra versao antiga;
- o modelo final fica inconsistente;
- a loss melhora localmente mas piora globalmente;
- ocorre esquecimento de fases anteriores.

Por isso SAINT-G precisa de ciclos de consolidacao.

## 10. Ciclos de Consolidacao

Depois de treinar varias partes, SAINT-G executa uma fase curta onde revisita partes importantes.

Exemplo:

```text
rodada 1:
  treina camada 0
  treina camada 1
  treina camada 2

consolidacao:
  revisita camada 0 com deltas das camadas 1 e 2 ativos
  revisita camada 1 com deltas das camadas 0 e 2 ativos
  revisita camada 2 com deltas das camadas 0 e 1 ativos
```

Isso tenta reduzir conflito entre atualizacoes.

## 11. Scheduler de Partes

SAINT-G precisa decidir a ordem de treino.

Estrategias possiveis:

### 11.1 Ordem Sequencial

Treina da primeira camada ate a ultima.

Simples, mas pode acumular erro.

### 11.2 Ordem Reversa

Treina da saida para a entrada.

Pode ser util quando a cabeca final precisa se adaptar primeiro.

### 11.3 Ordem por Sensibilidade

Calcula quais partes mais afetam a loss e prioriza essas partes.

Metricas possiveis:

- norma do gradiente;
- variacao da loss;
- magnitude da ativacao;
- estimativa Fisher;
- erro por camada.

### 11.4 Ordem Aleatoria Controlada

Escolhe partes aleatoriamente, mas garante cobertura.

Pode evitar vieses de ordem.

### 11.5 Curriculum por Tamanho

Comeca com partes maiores e depois refina blocos menores.

Exemplo:

```text
fase inicial: adapter por camada
fase media: submatrizes
fase final: blocos pequenos ou multi-escala
```

No modo multi-escala, SAINT-G pode testar varios tamanhos:

```text
64x64 -> 16x16 -> 8x8 -> 4x4 -> 2x2
```

Ele so desce para tamanhos menores se o ganho esperado justificar o custo.

### 11.6 Ordem por Reuso de Padrao

Prioriza grupos de blocos que aparecem muitas vezes.

Exemplo:

```text
padrao A aparece em 120.000 blocos
padrao B aparece em 2.000 blocos
padrao C aparece em 30 blocos
```

SAINT-G pode treinar primeiro o padrao A porque uma atualizacao nele afeta muitas posicoes ao mesmo tempo.

Essa estrategia e especialmente importante para o codebook.

### 11.7 Ordem por Ganho por Byte

SAINT-G pode ordenar partes por eficiencia:

```text
ganho_estimado_de_loss / bytes_de_memoria_treinavel
```

Isso combina qualidade e limite de VRAM.

Uma parte so entra no treino se justificar o custo de memoria.

### 11.8 Ordem por Erro de Reconstrucao

Quando uma regiao e aproximada por codebook, SAINT-G pode medir o erro de reconstrucao.

Regioes com erro alto podem ser refinadas:

```text
8x8 com erro baixo -> manter
8x8 com erro alto -> dividir em 4x4
4x4 com erro alto -> dividir em 2x2
```

Isso cria um scheduler adaptativo guiado por qualidade de representacao.

## 12. Orçamento de VRAM

SAINT-G e guiado por memoria.

O usuario define:

```text
vram_gb = 12
```

SAINT-G calcula:

- quantas partes podem estar ativas;
- tamanho de micro-batch;
- comprimento de sequencia;
- precision;
- offload;
- tamanho do optimizer state;
- margem de seguranca.

O objetivo nao e usar toda a VRAM, mas ficar abaixo de um limite seguro.

Exemplo:

```text
VRAM alvo: 12 GB
VRAM maxima permitida: 11.2 GB
margem: 0.8 GB
```

## 12.1 Orcamento por Camada

SAINT-G nao deve distribuir memoria de forma uniforme entre todas as camadas.

Nem toda camada, matriz ou submatriz tem a mesma importancia para uma tarefa.

Exemplo:

```text
camada 0: 2% dos blocos treinaveis
camada 8: 6% dos blocos treinaveis
camada 16: 10% dos blocos treinaveis
camada 24: 4% dos blocos treinaveis
camada final: 8% dos blocos treinaveis
```

O orcamento pode variar tambem por tipo de matriz:

```text
attention.Wq: 10%
attention.Wk: 5%
attention.Wv: 15%
attention.Wo: 10%
mlp.up: 25%
mlp.gate: 20%
mlp.down: 15%
```

A distribuicao pode ser guiada por:

- mapa de sensibilidade;
- erro de reconstrucao;
- ganho por byte;
- tipo de tarefa;
- historico de melhoria;
- custo de offload;
- estabilidade do gradiente.

Isso evita gastar VRAM em partes que quase nao contribuem.

## 12.2 Orçamento Adaptativo

O orcamento pode mudar durante o treino.

Exemplo:

```text
inicio:
  mais orcamento em camadas finais e lm_head

meio:
  mais orcamento em MLPs sensiveis

fim:
  refinamento em blocos pequenos de alta sensibilidade
```

SAINT-G pode realocar memoria quando perceber que uma regiao parou de melhorar.

Fluxo:

```text
1. treinar regioes selecionadas
2. medir ganho real
3. reduzir orcamento de regioes saturadas
4. aumentar orcamento de regioes promissoras
5. repetir
```

## 13. Offload

Para modelos grandes, SAINT-G precisa mover dados entre:

- GPU;
- RAM;
- disco/NVMe.

Possivel politica:

```text
parte ativa -> GPU
pesos congelados proximos -> GPU ou RAM
pesos distantes -> RAM/NVMe
optimizer state da parte ativa -> GPU/RAM
optimizer state inativo -> disco
```

Offload reduz VRAM, mas aumenta tempo.

## 13.1 Cache de Grupos de Blocos

Quando varios blocos compartilham o mesmo padrao, SAINT-G pode manter um cache de resultados intermediarios.

Exemplo:

```text
grupo G7:
  codebook_id = 7
  posicoes = [D12, D98, D301, ...]
```

Se o mesmo bloco efetivo aparece em muitas posicoes, o runtime pode:

```text
1. calcular a transformacao do grupo G7
2. reutilizar o resultado nas posicoes associadas
3. acumular gradientes de todas as posicoes no mesmo prototipo
```

Isso e diferente de apenas salvar memoria. A ideia e reduzir tambem trabalho repetido.

Limitacao: isso so ajuda se houver repeticao real ou repeticao imposta pela parametrizacao. Em matrizes densas totalmente livres, blocos exatamente iguais podem ser raros.

## 14. Reconstituicao Final

Ao final, SAINT-G combina:

```text
modelo base
+ deltas treinados
+ adapters
+ parametros de blocos
```

Dependendo do modo, ha duas opcoes.

### 14.1 Modelo Composto

Mantem o modelo base e aplica deltas durante o forward.

```text
W_eff = W_base + reconstruct(DeltaW)
```

Vantagem:

- checkpoints pequenos;
- reversivel;
- permite ligar/desligar deltas.

### 14.2 Modelo Fundido

Materializa os pesos finais:

```text
W_final = W_base + DeltaW
```

Vantagem:

- inferencia mais simples;
- nao precisa reconstruir deltas em tempo real.

Desvantagem:

- checkpoint final grande;
- perde modularidade.

## 15. Diferenca Para LoRA

LoRA aprende uma atualizacao de baixo rank:

```text
DeltaW = A B
```

SAINT-G pode aprender uma atualizacao por blocos:

```text
DeltaW = reconstruct(blocos_2x2)
```

Ou uma mistura:

```text
DeltaW = LoRA + BlockDelta
```

LoRA explora baixa dimensionalidade global.

SAINT-G por blocos exploraria estrutura local da matriz.

Essas tecnicas nao sao inimigas; podem ser combinadas.

Com codebook, SAINT-G tambem se diferencia por explorar repeticao:

```text
LoRA:
  DeltaW = A B

SAINT-G block-codebook:
  DeltaW[i, j] = escala_ij * codebook[id_ij]
```

LoRA pergunta:

```text
existe uma atualizacao global de baixo rank?
```

SAINT-G pergunta:

```text
existem padroes locais reutilizaveis dentro das matrizes?
```

## 16. Diferenca Para Treino Tradicional

Treino tradicional:

```text
todos os parametros ativos
loss global
gradiente global
optimizer state global
update global
```

SAINT-G:

```text
parte pequena ativa
loss global
gradiente local
optimizer state local
update parcial
consolidacao posterior
```

## 17. Hipotese de Pesquisa

A hipotese do SAINT-G e:

```text
Uma LLM grande pode ser adaptada ou parcialmente treinada
por atualizacoes pequenas, locais e recomponiveis,
desde que essas atualizacoes sejam avaliadas pela loss global
e revisadas por ciclos de consolidacao.
```

Uma segunda hipotese e:

```text
Matrizes grandes de LLMs podem conter padroes locais repetiveis
ou aproximaveis por um codebook pequeno de blocos.
Se esses padroes forem treinados uma vez e reutilizados em muitas posicoes,
o custo de adaptacao pode cair sem destruir a expressividade global.
```

## 18. Riscos Tecnicos

Principais riscos:

- convergencia ruim;
- conflito entre partes;
- tempo alto por offload;
- perda de qualidade ao recompor;
- blocos 2x2 serem expressivos demais ou de menos;
- overhead maior que o ganho;
- dificuldade de medir contribuicao local;
- instabilidade em modelos muito grandes;
- dependencia forte da ordem de treino.
- blocos realmente iguais serem raros em pesos nao quantizados;
- agrupamento aproximado destruir informacao importante;
- cache de blocos custar mais do que economiza;
- codebook pequeno demais limitar aprendizado;
- codebook grande demais perder vantagem de compressao;
- mascara esparsa ignorar partes que seriam importantes depois.

## 19. Experimentos Necessarios

Antes de tentar 70B, SAINT-G deve ser testado em modelos pequenos.

### Experimento 0 - Reconstrucao de Matriz

Antes de treinar uma LLM, SAINT-G deve provar que consegue representar matrizes.

Tarefa:

```text
matriz alvo W
  -> aproximar W com codebook multi-escala
  -> reconstruir W_aprox
  -> medir erro entre W e W_aprox
```

Comparar contra:

- SVD truncado;
- LoRA equivalente;
- quantizacao;
- blocos fixos sem codebook;
- codebook de tamanho unico;
- codebook multi-escala.

Metricas:

- erro de reconstrucao;
- numero de parametros;
- compressao;
- tempo de reconstrucao;
- memoria usada;
- erro por regiao;
- taxa de reutilizacao de prototipos.

Esse experimento separa o problema de representacao do problema de treino.

### Experimento 1 - Matriz Simples Treinavel

Treinar uma camada linear usando blocos `2x2`.

Comparar:

- matriz cheia;
- blocos completos;
- blocos diagonais;
- blocos rotacionais;
- blocos com codebook;
- blocos com assinaturas iguais/parecidas;
- LoRA.

Medir tambem:

- quantos blocos repetidos aparecem naturalmente;
- quanto erro aparece ao agrupar blocos parecidos;
- se calcular por grupo reduz tempo real;
- se o codebook aprende padroes reutilizaveis.

Tambem testar o roteador de blocos:

- roteador fixo;
- roteador por erro de reconstrucao;
- roteador por sensibilidade;
- roteador por ganho por byte;
- roteador multi-escala.

### Experimento 2 - Mini Transformer

Treinar um transformer pequeno com:

- treino tradicional;
- treino por camadas;
- treino por submatrizes;
- treino por blocos `2x2`;
- treino por blocos `2x2` com codebook;
- treino por mascara esparsa de sensibilidade;
- consolidacao.

### Experimento 2.1 - Mapa de Sensibilidade

Validar se o mapa de sensibilidade realmente seleciona partes uteis.

Comparar:

- blocos escolhidos aleatoriamente;
- blocos escolhidos por norma do gradiente;
- blocos escolhidos por impacto na loss;
- blocos escolhidos por frequencia de padrao;
- blocos escolhidos por ganho por byte.

### Experimento 3 - Modelo 3B

Adaptar modelo 3B com dataset pequeno.

Medir:

- loss;
- memoria;
- tempo;
- qualidade gerada;
- estabilidade.

### Experimento 4 - Modelo 14B

Escalar o metodo.

Verificar se o ganho de memoria compensa o overhead.

### Experimento 5 - Modelo 70B

Somente depois de validar os anteriores.

Objetivo:

- fine-tuning parcial;
- dataset pequeno;
- VRAM limitada;
- offload agressivo.

## 20. Medidas de Sucesso

SAINT-G deve ser comparado contra baselines.

Baselines:

- full fine-tuning em modelo pequeno;
- LoRA;
- QLoRA;
- adapters;
- treino por camadas sem blocos;
- treino head-only;
- blocos sem codebook;
- blocos com codebook;
- blocos aleatorios;
- blocos escolhidos por sensibilidade.

Metricas:

- loss final;
- tokens por segundo;
- VRAM maxima;
- RAM usada;
- tamanho do checkpoint;
- parametros treinaveis;
- qualidade em tarefas;
- degradacao apos merge;
- tempo ate melhoria mensuravel;
- numero de blocos ativos;
- taxa de reutilizacao de codebook;
- erro de reconstrucao dos blocos agrupados;
- ganho por byte de VRAM;
- ganho por parametro treinavel.

## 21. Estimativa de Aplicabilidade

### 3B

Bom alvo inicial.

Permite iterar, medir e comparar.

### 14B

Alvo intermediario.

Deve revelar gargalos de offload e scheduler.

### 70B

Alvo extremo.

Provavelmente viavel apenas como adaptacao parcial, com base quantizada, offload e dataset pequeno.

## 22. Pergunta Principal

A pergunta que SAINT-G precisa responder empiricamente:

```text
Atualizacoes locais, pequenas e recomponiveis conseguem produzir
melhoria global suficiente para justificar o overhead?
```

Se sim, SAINT-G pode virar um paradigma pratico para treino local limitado por VRAM.

Se nao, ainda pode produzir componentes uteis:

- memory planner;
- scheduler de blocos;
- adapters estruturados;
- analise de sensibilidade;
- codebook de blocos;
- cache de grupos repetidos;
- mascaras esparsas treinaveis;
- roteador de blocos;
- sistema de orcamento por camada;
- benchmark de reconstrucao de matrizes;
- ferramentas de treino parcial.

## 23. Resumo

SAINT-G propoe:

```text
modelo grande
  -> dividir em partes pequenas
  -> medir sensibilidade
  -> agrupar blocos iguais ou parecidos
  -> representar atualizacoes de forma compacta
  -> treinar poucas partes ou padroes por vez
  -> reutilizar calculos de grupos repetidos
  -> usar loss global
  -> salvar deltas parciais
  -> revisar conflitos
  -> recompor modelo final
```

O paradigma nao elimina o custo de passar dados pelo modelo, mas tenta reduzir o custo de manter todos os parametros treinaveis ao mesmo tempo.

O nucleo da ideia e:

```text
loss global, atualizacao local, recomposicao final
```

Com a melhoria de codebook, o nucleo expandido fica:

```text
loss global,
atualizacao local,
padroes compartilhados,
calculo agrupado,
recomposicao final
```

A formulacao mais completa do paradigma fica:

```text
SAINT-G = sparse multi-scale block-codebook delta training
```

Em portugues:

```text
treino de deltas esparsos por dicionario multi-escala de blocos
```
