# Criterios de Sucesso e Falha SAINT

Este documento define como avaliar se o SAINT esta funcionando ou falhando.

SAINT significa **Simple AI Node Training**.

Definicao:

```text
SAINT = sparse multi-scale block-codebook delta training
```

Em portugues:

```text
treino de deltas esparsos por dicionario multi-escala de blocos
```

## 1. Principio de Avaliacao

SAINT nao deve ser avaliado por promessa, mas por comparacao.

Toda fase precisa ter:

- baseline;
- metrica;
- criterio minimo;
- criterio de parada;
- registro reproduzivel.

Regra central:

```text
nao escalar antes de provar
```

## 2. Sucesso Conceitual

SAINT tem sucesso conceitual se a definicao ficar clara e testavel:

```text
loss global
atualizacao local
deltas esparsos
codebook multi-escala
roteamento por sensibilidade
recomposicao final
```

Falha conceitual se:

- o metodo nao se diferencia de LoRA/adapters;
- nao houver mecanismo claro de economia;
- nao houver forma de medir vantagem;
- a proposta depender de treinar todos os blocos.

## 3. Sucesso em Reconstrucao de Matriz

Sucesso se o codebook multi-escala conseguir pelo menos um:

- menor erro que codebook de tamanho unico com mesmo numero de parametros;
- menos parametros para erro similar;
- taxa de reutilizacao alta;
- compressao melhor que uma baseline simples;
- reconstrucao estavel em matrizes reais de modelos.

Falha se:

- erro for alto demais;
- SVD/LoRA/quantizacao simples vencerem em todos os casos;
- blocos nao forem reutilizados;
- multi-escala nao melhorar nada;
- custo de reconstruir for alto demais.

Metricas:

- erro L1;
- erro L2;
- erro relativo;
- compressao;
- taxa de reutilizacao;
- tempo de reconstrucao;
- memoria usada.

## 4. Sucesso em Blocos Reutilizaveis

Sucesso se:

- blocos iguais ou parecidos aparecem em quantidade util;
- clustering gera prototipos reutilizaveis;
- codebook treinavel reduz erro;
- agrupamento reduz parametros ou checkpoint.

Falha se:

- cada bloco precisar ser unico;
- codebook precisar ficar tao grande quanto a matriz;
- agrupamento destruir informacao critica;
- overhead do agrupamento superar economia.

## 5. Sucesso do Roteador

Sucesso se o roteador escolher regioes melhores que regras triviais.

Comparar contra:

- tudo congelado;
- tudo `2x2`;
- tudo `4x4`;
- selecao aleatoria;
- selecao uniforme por camada.

Sucesso minimo:

```text
roteador melhora qualidade ou reduz memoria
em relacao a pelo menos uma baseline simples.
```

Falha se:

- roteador escolher regioes sem ganho;
- planos forem instaveis;
- exigir ajustes manuais excessivos;
- ignorar regioes importantes.

## 6. Sucesso do Mapa de Sensibilidade

Sucesso se a selecao por sensibilidade superar selecao aleatoria.

Metricas:

- reducao de loss por bloco treinado;
- ganho por byte;
- ganho por parametro;
- estabilidade;
- repetibilidade entre batches.

Falha se:

- ranking for ruidoso demais;
- blocos escolhidos nao melhorarem loss;
- selecao aleatoria for igual ou melhor;
- custo de calcular sensibilidade for alto demais.

## 7. Sucesso em Camada Linear

Sucesso se SAINT aprender uma funcao alvo com:

- menos parametros que treino full;
- menor optimizer state;
- checkpoint menor;
- loss competitiva;
- ganho por byte melhor que alguma baseline.

Baselines:

- treino full;
- LoRA;
- bloco fixo;
- codebook tamanho unico;
- codebook multi-escala sem roteador.

Falha se:

- nao convergir;
- perder sempre para LoRA;
- codebook nao aprender;
- roteamento nao ajudar;
- deltas ficarem instaveis.

## 8. Sucesso em Mini-Transformer

Sucesso se:

- loss global com atualizacao local convergir;
- consolidacao reduzir conflito;
- SAINT superar head-only;
- SAINT se aproximar de LoRA em qualidade ou vencer em memoria;
- checkpoints forem menores que baselines relevantes.

Falha se:

- mini-transformer nao aprender;
- treino por partes piorar sempre;
- consolidacao nao ajudar;
- deltas causarem degradacao apos merge;
- ordem de treino dominar o resultado de forma imprevisivel.

## 9. Sucesso em Modelos Reais Pequenos

Sucesso se:

- SAINT roda de ponta a ponta;
- melhora loss ou perplexity;
- salva checkpoint recomponivel;
- compara com LoRA/QLoRA;
- nao degrada muito o modelo base;
- usa memoria dentro do planejado.

Falha se:

- OOM recorrente;
- sem melhoria mensuravel;
- overhead inviavel;
- qualidade muito abaixo de LoRA;
- merge quebra o modelo.

## 10. Sucesso em 3B

Sucesso minimo:

```text
rodar um treino parcial em 3B
com VRAM controlada
checkpoint recomponivel
e melhoria mensuravel de loss.
```

Sucesso forte:

- melhor ganho por byte que LoRA em algum experimento;
- checkpoint menor;
- boa estabilidade;
- consolidacao melhora resultado.

Falha:

- tempo impraticavel;
- sem ganho;
- OOM frequente;
- resultado sempre pior que baseline simples.

## 11. Sucesso em 14B

Sucesso minimo:

- estimativa de memoria correta;
- treino parcial roda;
- offload nao impede totalmente o experimento;
- deltas salvam e carregam;
- alguma melhoria aparece em dataset pequeno.

Falha:

- offload torna o treino inutilizavel;
- memory planner erra muito;
- qualidade nao melhora;
- overhead do codebook domina.

## 12. Sucesso em 70B

SAINT nao promete treino full de 70B.

Sucesso minimo em 70B:

```text
modelo base quantizado/congelado
deltas SAINT esparsos
micro-batch 1
offload agressivo
ciclo completo de treino parcial
sem OOM
checkpoint recomponivel
alguma melhoria mensuravel
```

Sucesso forte:

- comparavel a QLoRA em alguma metrica;
- checkpoint menor;
- menor VRAM efetiva;
- boa qualidade em tarefa especifica.

Falha:

- nao roda;
- roda mas nao aprende;
- leva tempo impraticavel;
- QLoRA domina em todos os aspectos;
- recomposicao degrada o modelo.

## 13. Criterios de Falha Global

Reavaliar o projeto se:

- codebook multi-escala nao comprimir matrizes;
- blocos reutilizaveis nao aparecem;
- sensibilidade nao supera aleatorio;
- roteador nao melhora nada;
- camada linear nao converge;
- mini-transformer nao converge;
- LoRA/QLoRA vencem sempre;
- overhead e maior que a economia;
- implementacao fica complexa demais para o ganho.

## 14. Criterios de Sucesso Global

SAINT e promissor se demonstrar pelo menos uma vantagem clara:

- menor memoria;
- menor checkpoint;
- melhor ganho por byte;
- melhor ganho por parametro treinavel;
- adaptacao local eficiente;
- codebook reutilizavel;
- recomposicao estavel;
- complementaridade com LoRA;
- capacidade de rodar em VRAM limitada onde treino full nao roda.

## 15. Metricas Padrao

Toda run deve registrar:

- loss;
- perplexity quando aplicavel;
- tokens/s;
- VRAM maxima;
- RAM maxima;
- parametros totais;
- parametros treinaveis;
- porcentagem treinavel;
- tamanho do checkpoint;
- taxa de reutilizacao de codebook;
- erro de reconstrucao;
- ganho por byte;
- ganho por parametro;
- tempo ate primeira melhoria;
- degradacao apos merge.

## 16. Baselines Obrigatorias

Por fase:

### Matriz

- SVD;
- LoRA equivalente;
- quantizacao;
- blocos fixos;
- codebook tamanho unico.

### Camada Linear

- treino full;
- LoRA;
- adapters simples;
- bloco fixo;
- codebook sem roteador.

### Transformer

- treino tradicional pequeno;
- head-only;
- LoRA;
- QLoRA quando aplicavel;
- adapters.

## 17. Criterio de Avanco Entre Fases

Avancar somente se:

```text
fase atual tem resultado reproduzivel
baseline foi rodada
metricas foram registradas
falhas foram documentadas
proxima fase tem hipotese clara
```

Nao avancar se:

- resultado depende de sorte;
- nao ha baseline;
- nao ha medicao de memoria;
- nao ha checkpoint;
- nao ha explicacao para ganho/perda.

## 18. Documento de Decisao

Ao fim de cada fase, criar uma decisao:

```text
continuar
ajustar
reduzir escopo
descartar hipotese
```

Formato recomendado:

```text
Fase:
Hipotese:
Baselines:
Resultado:
Metricas:
Falhas:
Decisao:
Proxima acao:
```

## 19. Resumo

SAINT so deve escalar se provar valor em pequeno.

O criterio final nao e "funcionou uma vez".

O criterio final e:

```text
funciona,
mede,
compara,
reproduz,
e justifica o custo.
```

