# Fase 18 - DRM Growth Protocol e Online Graft Search

Status: **pendente**.

## Objetivo

Se o DRM grafted passar em 125M/350M, o proximo passo nao deve ser apenas
aumentar o modelo. A Fase 18 transforma o resultado em uma linha cientifica
defensavel:

```text
DRM-Growth Protocol
```

e prepara o salto para:

```text
Distributed Validation-Gated Grafting
DRM-GOS: DRM Graft Orchestration System
```

## Tese

Modelos podem ganhar capacidade nova por enxertos estruturados, aceitos por
validacao e recomponiveis, com melhor eficiencia operacional que atualizacao
densa em certos regimes.

O foco deixa de ser apenas loss e passa a ser:

- replicacao;
- comparacao contra baselines fortes;
- retencao;
- controle;
- orquestracao distribuida de candidatos.

## Marco 1 - Replication Run

Objetivo:

Repetir os resultados positivos para separar sinal de sorte.

Entregas:

- varias seeds;
- outro split de dados;
- pelo menos uma config DRM diferente;
- mesma metrica de validacao;
- relatorio de variancia.

Criterio:

Passa se o ganho grafted aparecer de forma consistente em tres ou mais runs ou
se o relatorio explicar claramente quando ele falha.

## Marco 2 - Baselines Fortes

Objetivo:

Comparar SAINT-G contra controles que realmente competem.

Baselines:

- LoRA bem tunado;
- QLoRA quando couber;
- full-module linear;
- sparse/budgeted delta;
- adapters classicos quando aplicavel.

Criterio:

Passa se SAINT-G vencer em algum eixo relevante ou se ficar claro que a
vantagem esta em checkpoint, memoria, recomposicao ou controle, nao em loss
absoluta.

## Marco 3 - Retencao e Regressao

Objetivo:

Verificar se os grafts adicionam capacidade sem destruir o que o modelo sabia.

Metricas:

- old validation loss;
- new validation loss;
- catastrophic forgetting;
- merge/consolidation drift;
- checkpoint recomposition exactness.

Criterio:

Passa se enxertos aprovados nao criarem regressao clara em exemplos antigos e
se o checkpoint recomposto bater a validacao do runtime.

## Marco 4 - DRM-Growth Protocol

Objetivo:

Formalizar o metodo como protocolo cientifico:

```text
base DRM
-> sensitivity map
-> candidate grafts
-> validation-gated accept/reject/defer
-> freeze accepted grafts
-> staged consolidation
-> retention eval
```

Entregas:

- especificacao do protocolo;
- criterios de aceitacao;
- formato de artefato de graft;
- politica de freeze/consolidacao;
- criterios de regressao;
- tabela de falhas conhecidas.

Criterio:

Passa se outro experimento conseguir seguir o protocolo sem decisoes implicitas.

## Marco 5 - Safety e Control Evals

Objetivo:

Medir se a tese de crescimento controlado e observavel.

Avaliacoes:

- comportamento anomalo apos graft;
- separacao factual/incerto/arriscado;
- robustez a prompts adversariais;
- tendencia a overfit ou memorizar;
- capacidade de desligar/remover grafts sem quebrar o modelo;
- rastreabilidade: qual graft mudou qual comportamento.

Criterio:

Passa se as avaliacoes produzirem metricas comparaveis entre:

```text
base DRM
full DRM
DRM grafted
```

## Marco 6 - Ponte 1.3B

Objetivo:

Escalar para 1.3B antes de 70B.

Motivo:

1.3B e grande o bastante para revelar gargalos reais, mas pequeno o bastante
para diagnosticar falhas antes de um salto caro.

Entregas:

- memory planner 1.3B;
- load/forward smoke;
- graft inference-only;
- treino parcial pequeno;
- comparacao contra LoRA/QLoRA se couber;
- checkpoint recomponivel.

Criterio:

Passa se 1.3B rodar ciclo parcial sem OOM e preservar as propriedades do
protocolo.

## Marco 7 - DRM-GOS: Cluster-Scale Online Graft Search

Objetivo:

Projetar busca online distribuida de enxertos.

Arquitetura:

```text
base DRM congelado
        |
        v
coordenador central
        |
        +--> worker GPU 1: testa graft A em layer 3
        +--> worker GPU 2: testa graft B em layer 7
        +--> worker GPU 3: testa graft C em FFN/down_proj
        +--> worker GPU 4: roda LoRA baseline/controle
        |
        v
validador central
        |
        v
accept / reject / defer / retrain
        |
        v
checkpoint composto recomponivel
```

Fluxo:

1. O modelo base fica congelado e versionado.
2. Cada worker recebe checkpoint base, shard de dados, alvo, budget, seed e
   criterios de parada.
3. Cada worker treina um candidato localmente.
4. O worker envia apenas o artefato do graft.
5. O validador recompõe `base + accepted_grafts + candidate`.
6. O coordenador decide aceitar, rejeitar, adiar ou retreinar com outro budget.

Regra critica:

```text
loss(base + accepted_grafts + candidate)
<
loss(base + accepted_grafts)
```

Restricoes:

- `new_gain > threshold`;
- `old_regression < limit`;
- `safety_regression < limit`;
- `checkpoint_growth < budget`;
- `conflict_score < limit`.

Criterio:

Passa se houver especificacao suficiente para implementar um prototipo com
workers independentes e validador central.

## Criterio de Sucesso da Fase

Sucesso minimo:

```text
replicacao multiseed
comparacao contra baselines fortes
protocolo DRM-Growth documentado
safety/control eval inicial
plano DRM-GOS especificado
```

Sucesso forte:

```text
resultado replicado em 125M/350M
ponte 1.3B executada
grafts aprovados por validacao central
retencao e safety sem regressao clara
```

## Proximo Passo Imediato

Se a Fase 17 passar:

1. Rodar replication run multiseed.
2. Adicionar LoRA/QLoRA forte no mesmo corpus.
3. Criar `docs/protocols/drm_growth_protocol.md`.
4. Criar spec inicial `docs/architecture/drm_gos.md`.
