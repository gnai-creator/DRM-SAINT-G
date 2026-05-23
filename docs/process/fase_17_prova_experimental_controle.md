# Fase 17 - Prova Experimental Publicavel e Controle

Status: **pendente**.

## Objetivo

Fechar a ponte entre `drm_transformer` e `SAINT-G` como uma prova
experimental publicavel:

```text
DRM full controlado vs DRM pequeno + enxertos SAINT-G
```

A pergunta central nao e se SAINT-G vence treino full em loss absoluta. A
pergunta correta e:

```text
um DRM pequeno + enxertos consegue se aproximar de um DRM full maior
com melhor eficiencia de memoria, checkpoint, tempo e controle?
```

Esta fase transforma a Fase 16 em evidencia cientifica mais completa. Antes de
70B, o projeto precisa demonstrar qualidade, eficiencia e controle em 125M/350M.

## Posicionamento

Depois de:

- `drm_transformer`: arquitetura geometrica;
- `SAINT-G`: crescimento por enxertos;

o proximo produto cientifico deve ser:

```text
structured grafting for controlled model growth
```

Nao vender como:

```text
treino full barato
substituto universal de LoRA/QLoRA
70B do zero em GPU domestica
```

Mas sim como:

```text
crescimento controlado por enxertos recomponiveis
com comparacao multi-eixo contra full DRM controlado
```

## Entregas

- benchmark final `DRM full 125M` vs `DRM 5M + grafts`;
- repetir com `DRM full 350M` se o baseline estiver viavel;
- relatorio com loss, perplexity, VRAM, tempo, checkpoint e ganho por parametro;
- avaliacao de retencao/regressao apos enxertos;
- avaliacao simples de controle/safety;
- tabela de criterios de sucesso/falha;
- recomendacao tecnica para Fase 18.

## Marco 1 - Benchmark Full vs Grafted

Objetivo:

Comparar em condicoes honestas:

```text
DRM full 125M
DRM 5M + SAINT-G grafted ate budget/capacidade comparavel
```

Metricas:

- validation loss;
- perplexity;
- tokens/s;
- CUDA peak;
- tempo total;
- checkpoint bytes;
- parametros treinaveis;
- ganho por parametro treinavel;
- recomposicao de checkpoint;
- distancia ate full 125M.

Criterio:

Passa se o relatorio deixa claro:

- onde grafted ganha;
- onde grafted perde;
- se o ganho operacional justifica continuar.

## Marco 2 - Retencao e Regressao

Objetivo:

Medir se os enxertos melhoram o conjunto novo sem destruir exemplos antigos.

Entregas:

- split antigo/novo;
- loss antes/depois em exemplos antigos;
- loss antes/depois em exemplos de treino/validacao;
- regressao maxima aceitavel;
- tabela por enxerto aprovado.

Criterio:

Passa se a melhoria de validacao nao vier acompanhada de regressao clara em
exemplos antigos.

## Marco 3 - Avaliacao de Controle

Objetivo:

Criar uma primeira camada `DRM-Control Eval`.

Medir se a geometria/enxertos ajudam em:

- deteccao de estados anômalos;
- separacao entre factualidade, incerteza e risco;
- regressao apos enxertos;
- comportamento enganoso/deceptivo simples em tarefas toy;
- corrigibilidade/shutdown em tarefas toy;
- retencao de capacidades antigas;
- interpretabilidade de anchors/metrica quando disponivel.

Criterio:

Passa se pelo menos uma avaliacao de controle roda de ponta a ponta e produz
metrica comparavel entre:

```text
base 5M
grafted
full 125M
```

## Marco 4 - Relatorio Publicavel

Objetivo:

Transformar os resultados em um relatorio publicavel.

Entregas:

- resumo executivo;
- metodologia reproduzivel;
- tabelas principais;
- limitacoes;
- comparacao honesta contra full;
- comparacao operacional contra grafted;
- recomendacao para escala 350M/70B;
- texto base para README, LinkedIn, Hugging Face e paper curto.

Criterio:

Passa se o projeto tiver resposta clara para:

```text
SAINT-G deve avancar para 350M/70B?
Qual baseline carregar para a Fase 18?
O foco deve ser qualidade, escala ou infraestrutura?
```

## Criterio de Sucesso da Fase

Sucesso minimo:

```text
benchmark full vs grafted reproduzivel
checkpoint grafted recomponivel
metricas multi-eixo registradas
avaliacao de controle inicial executada
```

Sucesso forte:

```text
DRM 5M + grafts demonstra vantagem clara em pelo menos um eixo operacional
sem regressao forte de retencao
e com sinal positivo em controle/safety toy
```

Falha:

- grafted nao melhora a base;
- checkpoint nao recompõe;
- eficiencia operacional nao compensa o gap de qualidade;
- avaliacao de controle nao diferencia nenhum regime;
- full 125M/350M nao fica comparavel por dados ou tokenizacao.

## Proximo Passo Imediato

Consolidar os resultados da Fase 16 em uma tabela unica:

```text
base 5M
full 125M smoke
grafted 4F
grafted 4G
grafted 4H
grafted 4I
```

Depois, executar o Marco 1 com o melhor checkpoint grafted recomponivel.
