# Hipoteses de Pesquisa SAINT-G

Este documento define as hipoteses que o SAINT-G precisa testar.

SAINT-G significa **DRM por Enxerto com SAINT-G-Phi**.

Definicao:

```text
SAINT-G = sparse multi-scale block-codebook delta training
```

Em portugues:

```text
treino de deltas esparsos por dicionario multi-escala de blocos
```

## 1. Hipotese Principal

Uma LLM grande pode ser adaptada parcialmente usando deltas esparsos, locais e recomponiveis, desde que:

- a loss seja global;
- a atualizacao seja local;
- as regioes treinaveis sejam escolhidas por sensibilidade;
- os deltas sejam representados por blocos multi-escala;
- padroes repetidos sejam compartilhados por codebooks;
- ciclos de consolidacao reduzam conflito entre partes.

Forma curta:

```text
loss global + atualizacao local + deltas recomponiveis
podem adaptar modelos grandes com menos memoria.
```

## 2. Hipotese de Representacao

Matrizes grandes de modelos neurais podem ser aproximadas por uma combinacao de:

- blocos multi-escala;
- codebooks pequenos;
- escalas por regiao;
- refinamentos locais;
- blocos livres apenas em regioes criticas.

Pergunta:

```text
Um codebook multi-escala representa matrizes reais de modelos
com erro competitivo e menos parametros?
```

Teste minimo:

```text
W -> codebook multi-escala -> W_aprox
```

Comparar contra:

- SVD truncado;
- LoRA equivalente;
- quantizacao;
- blocos fixos;
- codebook de tamanho unico.

## 3. Hipotese de Reutilizacao

Mesmo que blocos exatamente iguais sejam raros, blocos parecidos podem ser agrupados de forma util.

SAINT-G espera encontrar ou induzir padroes reutilizaveis.

Perguntas:

- Existem blocos naturalmente parecidos nas matrizes?
- A quantizacao aumenta reutilizacao sem destruir qualidade?
- Clustering de blocos produz prototipos uteis?
- O codebook treinavel aprende padroes reutilizaveis?

Se nao houver reutilizacao relevante, a vantagem do codebook cai.

## 4. Hipotese Multi-Escala

Blocos de um unico tamanho nao sao ideais para todas as regioes.

Hipotese:

```text
regioes simples podem usar blocos grandes;
regioes complexas precisam de blocos pequenos;
regioes criticas podem precisar de delta livre ou LoRA auxiliar.
```

O codebook multi-escala deve superar:

- `2x2` puro;
- `4x4` puro;
- blocos fixos sem roteamento.

## 5. Hipotese de Sensibilidade

Nem todas as partes do modelo contribuem igualmente para a melhoria da loss.

Hipotese:

```text
um mapa de sensibilidade escolhe regioes melhores que selecao aleatoria.
```

Metodos candidatos:

- norma do gradiente;
- gradiente vezes peso;
- impacto por mascaramento;
- estimativa Fisher;
- magnitude de ativacao;
- erro de reconstrucao;
- ganho por byte.

Esta hipotese e essencial. Sem ela, SAINT-G pode gastar computacao em blocos irrelevantes.

## 6. Hipotese do Roteador

Um roteador heuristico consegue escolher uma representacao adequada por regiao.

Exemplo:

```text
erro baixo + baixa sensibilidade  -> congelar ou bloco grande
erro alto + baixa sensibilidade   -> bloco medio
erro baixo + alta sensibilidade   -> bloco pequeno
erro alto + alta sensibilidade    -> bloco pequeno + LoRA/delta livre
```

Pergunta:

```text
roteamento melhora o tradeoff entre qualidade e memoria?
```

## 7. Hipotese de Ganho por Byte

O melhor uso de VRAM nao e treinar o maior numero de parametros, mas treinar as partes com maior retorno por memoria usada.

Metrica:

```text
ganho_de_loss / bytes_treinaveis
```

Hipotese:

```text
ordenar partes por ganho por byte melhora eficiencia.
```

## 8. Hipotese de Orcamento por Camada

Distribuir memoria igualmente entre camadas e matrizes e ineficiente.

Hipotese:

```text
orcamento adaptativo por camada e tipo de matriz melhora qualidade
para o mesmo limite de VRAM.
```

Exemplo:

```text
algumas MLPs podem merecer mais orcamento;
algumas matrizes de attention podem merecer menos;
camadas finais podem ser mais importantes em certas tarefas.
```

## 9. Hipotese de Consolidacao

Treinar partes separadas cria conflitos.

Hipotese:

```text
ciclos curtos de consolidacao reduzem conflito entre deltas
e melhoram a loss global apos recomposicao.
```

Sem consolidacao, o modelo pode melhorar localmente e piorar globalmente.

## 10. Hipotese de Checkpoint Pequeno

Deltas SAINT-G podem ser muito menores que checkpoints completos.

Hipotese:

```text
codebooks + ids + escalas + mascaras
podem salvar adaptacoes uteis com checkpoint pequeno.
```

Comparar contra:

- checkpoint completo;
- LoRA;
- QLoRA;
- adapters.

## 11. Hipotese de Cache de Grupos

Quando varios blocos compartilham prototipos, e possivel reduzir trabalho repetido.

Hipotese:

```text
agrupar blocos por codebook_id reduz custo de aplicacao
ou custo de gradiente em alguns regimes.
```

Risco:

```text
o overhead de agrupamento pode ser maior que o ganho.
```

## 12. Hipotese de Escala

Se SAINT-G funcionar em matriz isolada, camada linear e mini-transformer, pode ser escalado gradualmente.

Escala planejada:

```text
matriz isolada
camada linear
mini-transformer
modelo pequeno
3B
14B
70B
```

Hipotese:

```text
os ganhos de memoria continuam relevantes ao aumentar o modelo,
mesmo que o tempo aumente por offload e roteamento.
```

## 13. Hipotese Contra LoRA

SAINT-G nao precisa vencer LoRA em tudo.

Hipotese realista:

```text
SAINT-G pode vencer ou complementar LoRA em regimes especificos:
- menor checkpoint;
- melhor ganho por byte;
- melhor adaptacao local;
- maior reutilizacao estrutural;
- combinacao SAINT-G + LoRA.
```

Se SAINT-G sempre perder para LoRA/QLoRA em qualidade, memoria e tempo, o paradigma deve ser reavaliado.

## 14. Hipoteses Nulas

SAINT-G pode falhar se:

- matrizes reais nao tiverem padroes reutilizaveis suficientes;
- codebook multi-escala nao comprimir bem;
- sensibilidade nao escolher regioes uteis;
- roteador gerar planos ruins;
- atualizacoes locais nao acumularem melhoria global;
- consolidacao nao resolver conflitos;
- overhead superar economia;
- LoRA/QLoRA forem melhores em todos os regimes.

Essas hipoteses nulas devem ser tratadas como testes reais, nao como detalhes secundarios.

## 15. Ordem de Validacao

As hipoteses devem ser testadas nesta ordem:

```text
1. representacao de matriz
2. reutilizacao de blocos
3. codebook multi-escala
4. roteador
5. treino de camada linear
6. mapa de sensibilidade
7. mini-transformer
8. consolidacao
9. modelos reais pequenos
10. escala 3B/14B/70B
```

## 16. Resultado Esperado

SAINT-G sera promissor se as hipoteses mostrarem que:

- deltas esparsos aprendem algo util;
- codebooks reduzem memoria ou checkpoint;
- sensibilidade supera escolha aleatoria;
- multi-escala supera tamanho unico;
- consolidacao reduz conflito;
- a qualidade fica competitiva contra baselines.

SAINT-G deve ser descartado ou reposicionado se esses pontos nao aparecerem em escala pequena.

