# Fase 19 - Escala 70B

Status: **pendente**.

## Objetivo

Validar SAINT-G como adaptacao extrema em hardware limitado.

Esta fase nao tenta provar que conseguimos treinar um DRM 70B do zero em uma GPU
domestica. O objetivo e mais restrito e mais honesto:

```text
usar grafting/enxertos SAINT-G sobre um modelo grande congelado
com memoria controlada
checkpoint recomponivel
e melhoria mensuravel de loss em uma tarefa pequena
```

## Premissa Central

Sim: a Fase 19 deve usar **grafting**.

Mas nao e:

```text
construir um DRM 70B do zero
treinar todos os parametros
comparar contra full fine-tuning 70B
```

E:

```text
modelo base grande congelado
pesos base quantizados/offloadados
enxertos Phi/deltas pequenos
loss global
atualizacao local em poucos alvos
merge/checkpoint recomponivel
```

Como nao existe baseline full DRM 70B viavel no hardware alvo, a comparacao da
Fase 19 deve ser contra controles possiveis.

## O Que Comparar

Nao da para comparar contra:

- treino full de DRM 70B;
- full fine-tuning de todas as camadas;
- baseline denso completo de mesmo tamanho.

Comparacoes viaveis:

- base congelada sem enxerto;
- enxerto desligado, para detectar ruido experimental;
- SAINT-G/Phi em budgets diferentes;
- QLoRA/LoRA quando couber no mesmo limite de VRAM;
- full-module pequeno em modelos DRM menores, por extrapolacao;
- resultados das fases 5C/5D como referencia de comportamento em escala menor;
- tempo, memoria e tamanho de checkpoint contra controles sem treino.

## Condicoes

Nao e treino full de 70B.

E:

```text
modelo base quantizado/congelado
deltas SAINT-G esparsos
codebook multi-escala quando aplicavel
offload agressivo
micro-batch 1
dataset pequeno
loss global
atualizacao local
checkpoint recomponivel
```

## Baseline de Entrada

Antes de qualquer tentativa 70B, a fase deve continuar a partir de:

```text
drm_transformer/configs/scaling/multilingual/5m.yaml
```

Esse baseline e a ponte validada no Marco 5D. A Fase 19 nao deve saltar direto
para 70B sem manter uma trilha reproduzivel em escalas menores.

## Sequencia de Escala

A ordem recomendada e:

```text
5M -> proximo scaling disponivel -> smoke grande -> 70B congelado
```

Se nao houver configs intermediarias no `drm_transformer`, criar configs de
scaling progressivas antes do 70B:

```text
5M
25M
100M
500M
1B
3B
14B
70B
```

Cada salto so deve ocorrer se o anterior demonstrar:

- load controlado;
- forward controlado;
- enxerto aplicado;
- checkpoint salvo;
- merge/eval recomponivel;
- alguma melhoria de loss ou diagnostico claro de falha.

## Marcos

### Marco 1 - Planejamento de Memoria

Objetivo:

Estimar se o caminho 70B e executavel no hardware alvo antes de carregar o modelo.

Entregas:

- estimativa por dtype: `float16`, `bfloat16`, `int8`, `4bit`;
- estimativa de pesos congelados;
- estimativa de KV/cache quando houver geracao;
- estimativa de ativacoes com micro-batch 1;
- estimativa de parametros treinaveis Phi;
- estimativa de estado do otimizador;
- plano de offload CPU/GPU;
- limite de abortar antes de OOM.

Criterio:

Passa se o planner produzir uma decisao clara:

```text
run permitido
run permitido apenas com offload
run abortado antes de carregar
```

### Marco 2 - Load Congelado

Objetivo:

Carregar o modelo grande congelado sem treino.

Entregas:

- carregar pesos quantizados/offloadados;
- medir memoria apos load;
- rodar forward simples;
- medir loss base em dataset pequeno;
- salvar relatorio de load.

Criterio:

Passa se:

- modelo carrega sem OOM;
- forward roda;
- loss base e registrada;
- memoria real fica dentro do budget.

### Marco 3 - Enxerto Inference-Only

Objetivo:

Aplicar enxerto SAINT-G/Phi sem treino.

Entregas:

- inserir hook/enxerto em uma matriz alvo;
- rodar forward com enxerto zero;
- rodar forward com enxerto inicializado;
- validar rollback/desligamento do enxerto;
- medir overhead de memoria e tempo.

Criterio:

Passa se:

- loss com enxerto zero bate com base dentro de tolerancia;
- enxerto pode ser ligado/desligado sem corromper modelo;
- overhead e mensurado.

### Marco 4 - Treino Parcial Minimo

Objetivo:

Treinar poucos parametros SAINT-G/Phi em 70B congelado.

Entregas:

- escolher um alvo pequeno;
- micro-batch 1;
- poucos steps;
- optimizer somente para parametros do enxerto;
- medir loss antes/depois;
- salvar checkpoint do enxerto;
- validar resume.

Criterio:

Passa se:

- nao ha OOM;
- checkpoint salva e carrega;
- loss nao piora de forma sistematica;
- ao menos um run mostra melhoria mensuravel.

### Marco 5 - Merge e Reconstituicao

Objetivo:

Provar que o enxerto e recomponivel.

Entregas:

- salvar payload do enxerto;
- salvar estado minimo do otimizador quando aplicavel;
- aplicar merge temporario ou hook recomponivel;
- avaliar loss apos reload;
- validar checksum.

Criterio:

Passa se:

```text
train -> save -> validate -> resume -> merge/eval
```

funciona sem depender de estado em memoria.

### Marco 6 - Comparacao Contra Controles Viaveis

Objetivo:

Comparar contra o que realmente cabe no hardware.

Controles:

- base congelada;
- enxerto zero;
- SAINT-G/Phi;
- LoRA/QLoRA se couber;
- run menor 5M/escala intermediaria com full-module, como referencia.

Metricas:

- validation loss antes/depois;
- ganho por parametro treinavel;
- memoria maxima;
- tempo por step;
- checkpoint bytes;
- estabilidade por seed;
- regressao em exemplos antigos.

Criterio:

Passa se SAINT-G demonstrar vantagem em pelo menos um eixo relevante:

- menor memoria que LoRA/QLoRA;
- menor checkpoint;
- ganho positivo onde base congelada nao muda;
- maior ganho por parametro;
- melhor estabilidade em budget muito pequeno.

### Marco 7 - Decisao de Escala

Objetivo:

Decidir se 70B e um alvo valido ou se a proxima fase deve ficar em 14B/menor.

Entregas:

- relatorio final;
- tabela de memoria;
- tabela de qualidade;
- recomendacao tecnica;
- lista de bloqueios.

Criterio:

Passa se houver resposta clara:

```text
continuar 70B
voltar para 14B/3B
melhorar runtime/offload antes de prosseguir
```

## Experimentos

- estimativa de memoria;
- inferencia com deltas;
- fine-tuning pequeno;
- comparacao contra QLoRA quando possivel;
- medicao de tempo real;
- analise de qualidade;
- retencao em exemplos antigos;
- reconstituicao de checkpoint;
- sweep pequeno de budgets Phi;
- medicao de overhead do hook/enxerto.

## Criterio de Sucesso

O objetivo minimo:

```text
rodar um ciclo completo de treino parcial em 70B
sem OOM
com checkpoint recomponivel
e alguma melhoria mensuravel na loss
```

Sucesso forte:

```text
SAINT-G/Phi melhora loss
com checkpoint menor que LoRA/QLoRA
e memoria menor ou comparavel
mantendo estabilidade em mais de uma seed
```

Falha:

- OOM antes do forward;
- forward so funciona sem enxerto;
- treino parcial nao salva/recarrega;
- melhoria de loss nao aparece em nenhum regime;
- overhead de tempo torna o metodo impraticavel;
- QLoRA vence em todos os eixos relevantes.

## Riscos

- O modelo 70B pode nao caber nem quantizado com offload aceitavel.
- O custo de I/O pode dominar o tempo.
- Phi pode precisar de alvos diferentes em escala grande.
- Loss em dataset pequeno pode ser ruidosa.
- Sem baseline full 70B, a avaliacao cientifica precisa ser multi-eixo.
- Hooks podem gerar overhead maior que o ganho de compressao.
- Merge denso pode ser impossivel; pode ser necessario manter merge virtual.

## Decisao Sobre Comparacao

Como nao ha como montar e treinar um DRM 70B full do zero, a Fase 19 nao deve
exigir essa comparacao.

O baseline cientifico correto e:

```text
base congelada vs base + SAINT-G/Phi
```

E, quando possivel:

```text
base + SAINT-G/Phi vs base + QLoRA/LoRA
```

Para qualidade absoluta, usar modelos menores como controle:

```text
5M/menor com full-module
5M/menor com Phi
70B com Phi
```

Assim a pergunta muda de:

```text
SAINT-G vence treino full 70B?
```

para:

```text
SAINT-G permite adaptar um modelo grande congelado
com menos memoria e checkpoint recomponivel?
```

Essa e a pergunta certa para hardware limitado.

## Proximo Passo Imediato

Antes de qualquer carga 70B:

1. Criar memory planner especifico da Fase 19.
2. Rodar `multilingual/5m.yaml` com o mesmo pipeline de checkpoint/reload.
3. Adicionar retencao para as variantes Phi.
4. Medir CUDA real em 5M.
5. So entao tentar um smoke maior.
