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
Fase atual: Fase 7 - Runtime SAINT
Fase anterior: Fase 6 concluida
Proximo marco: Fase 7 - Runtime SAINT
```

Resumo do estado:

| Fase | Nome | Status |
|---|---|---|
| 0 | Fundacao Conceitual | Concluida |
| 1 | Biblioteca de Blocos | Concluida |
| 2 | Benchmark de Reconstrucao | Concluida |
| 3 | Roteador de Blocos | Concluida |
| 4 | Treino de Camada Linear | Concluida |
| 5 | Mini-Transformer | Concluida |
| 6 | Mapa de Sensibilidade | Concluida |
| 7+ | Escala e runtime completo | Pendente |

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

Status: **concluida**.

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

- pacote inicial `saint` - concluido;
- modulo `blocks` - concluido;
- funcao para particionar matrizes - concluido;
- funcao para reconstruir matrizes - concluido;
- suporte a padding - concluido;
- suporte a blocos de borda - concluido;
- calculo de assinaturas - concluido;
- agrupamento exato - concluido;
- agrupamento aproximado por quantizacao - concluido;
- metricas de reconstrucao - concluido;
- metricas de reutilizacao - concluido;
- codebook fixo inicial - concluido;
- testes unitarios - concluido;
- documentacao da fase em `docs/process/fase_1_biblioteca_blocos.md` - concluido.

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

Status: **concluida**.

### Objetivo

Testar se o codebook multi-escala representa matrizes melhor que alternativas simples.

### Entregas Iniciais

- modulo `saint.reconstruction`;
- geradores de matrizes sinteticas;
- baselines de quantizacao, low-rank, block-codebook e multi-scale simples;
- runner de benchmark;
- testes unitarios;
- benchmark real com matrizes do `drm_transformer`;
- documento `docs/process/fase_2_benchmark_reconstrucao.md`.

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

### Resultado

O benchmark diferencia regimes:

- matriz gaussiana nao mostra reuso relevante;
- matriz low-rank favorece baseline low-rank;
- matriz sparse mostra reuso moderado;
- matriz com blocos repetidos favorece codebook de blocos.
- matrizes reais do `drm_transformer` favorecem `block_codebook_4` como tradeoff inicial.

Veredito:

```text
hipotese parcialmente suportada;
avancar para Fase 3 com foco em roteamento por regiao.
```

### Criterio de conclusao

Prosseguir somente se o codebook multi-escala mostrar vantagem clara em pelo menos um destes pontos:

- menor erro com mesmo numero de parametros;
- menos parametros com erro parecido;
- alta taxa de reutilizacao;
- reconstrucao eficiente o suficiente para uso em treino.

## 4. Fase 3 - Roteador de Blocos

Status: **concluida**.

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

### Resultado

Implementado:

- `saint.routing`;
- roteador por erro de reconstrucao;
- roteador por score de orcamento;
- roteador com `freeze/zero_delta`;
- sensibilidade proxy por norma L1 da regiao;
- orcamento duro por metodo;
- score ponderado por sensibilidade;
- codebook com escala por bloco;
- codebook residual;
- baseline `routed_codebook`;
- baseline `routed_budget_first`;
- baseline `routed_sensitivity_budget`;
- testes unitarios;
- comparacao contra `block_codebook_4` e `hierarchical_codebook`.

Resultado em matrizes reais:

```text
routed_quality_first reduz erro medio para 0.0003,
mas compressao media fica 0.94.

routed_budget_first preserva erro medio 0.0028,
mas compressao media fica 0.97 e ainda nao atinge alvo 1.1.
```

A busca de `parameter_weight` foi adicionada:

```text
routed_budget_search erro medio 0.0080,
compressao media 1.01,
criterio automatico ainda falha para alvo 1.1.
```

O sweep de `quantization_step` encontrou o primeiro ponto aprovado:

```text
routed_budget_first
quantization_step: 0.0869
erro medio: 0.0991
compressao media: 1.1023
```

Esse resultado passa o criterio minimo configurado:

```text
erro relativo medio <= 0.1
compressao media >= 1.1
```

O roteador com freeze e sensibilidade comprime muito mais:

```text
routed_sensitivity_budget erro medio 0.7441,
compressao media 7.73.
```

Esse resultado nao e suficiente para aprovar a tecnica em reconstrucao de peso
bruto, mas e coerente com a proposta de SAINT para treino de deltas: congelar
uma regiao deve significar nao aplicar delta, mantendo o peso base intacto.

Tambem foram testados codebook com escala por bloco e codebook residual. Em
matrizes reais 64x64, eles ainda nao melhoraram o criterio automatico.

Conclusao:

```text
roteador melhora qualidade;
orcamento melhora compressao de forma limitada;
Fase 3 passa no criterio minimo de reconstrucao;
freeze melhora compressao, mas precisa ser validado em treino de deltas na Fase 4.
```

## 5. Fase 4 - Treino de Camada Linear

Status: **concluida**.

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

### Resultado Inicial

Foi criado o primeiro benchmark dependency-free de aprendizado:

```text
scripts/benchmark_linear_phase4.py
```

Configuracao inicial:

```text
y = W_target x
W_target = W_base + delta_target
W_base congelada
```

Resultado inicial:

```text
full_delta test_loss: 0.0000003, params: 64
saint_routed_delta test_loss: 0.0004701, params: 56
lora_rank_2 test_loss: 0.0034705, params: 32
sparse_sensitivity_delta test_loss: 0.0010289, params: 16
```

Leitura:

```text
SAINT ainda nao bate full_delta em loss,
mas aprende com menos parametros e supera LoRA rank 2 nesse caso sintetico.
```

A Fase 4 continua em andamento porque falta sweep de sementes, ranks de LoRA,
orcamentos do roteador e criterio automatico de sucesso/falha.

### Resultado do Sweep

Foi adicionado sweep com 5 sementes, LoRA rank 1/2/4 e tres orcamentos SAINT.

Melhor variante SAINT:

```text
saint_routed_f50_c25
test_loss medio: 0.0005749
params medios: 51.2
ganho/parametro: 0.00007862
```

Comparacao contra LoRA rank 2:

```text
lora_rank_2 test_loss medio: 0.0035034
lora_rank_2 params medios: 32.0
lora_rank_2 ganho/parametro: 0.00003379
```

O criterio automatico inicial foi:

```text
loss_ratio <= 1.0
parameter_ratio <= 2.0
gain_per_parameter_ratio >= 1.0
```

Resultado:

```text
saint_routed_f50_c25 passou contra lora_rank_2.
saint_routed_f25_c50 passou contra lora_rank_2.
saint_routed_f25_c25 falhou contra lora_rank_1 por exceder 2x parametros.
```

A Fase 4 continua em andamento porque falta testar tamanhos maiores, deltas
menos favoraveis a codebook e LoRA com ajustes mais amplos de hiperparametros.

### Resultado do Sweep de Regimes

Foi adicionado um sweep com:

```text
sizes: 8x8, 16x16, 32x32
delta_modes: repeated, dense
seeds: 11, 12
```

Resultado agregado:

```text
saint_routed_f50_c25 test_loss medio: 0.0043511
saint_routed_f50_c25 params medios: 364.0
lora_rank_2 test_loss medio: 0.0098293
lora_rank_2 params medios: 74.7
budgeted_full_delta_for_saint_routed_f50_c25 test_loss medio: 0.0035874
```

Leitura:

```text
SAINT ainda vence LoRA rank 2 em loss,
mas perde o criterio de parametros em 16x16 e 32x32.
Contra full_delta esparso com orcamento equivalente, SAINT ainda perde.
```

Conclusao atual:

```text
Fase 4 permanece em andamento.
O proximo trabalho e reduzir o crescimento de parametros do roteador
e compartilhar codebooks em escala de camada.
```

### Resultado do Codebook Global

Foi adicionada a variante:

```text
saint_global_capped
```

Ela usa codebook global por camada, limite de regioes livres e limite global de
prototipos.

Resultado agregado:

```text
saint_global_capped test_loss medio: 0.0074230
saint_global_capped params medios: 126.7
lora_rank_2 test_loss medio: 0.0098293
lora_rank_2 params medios: 74.7
budgeted_full_delta_for_saint_global_capped test_loss medio: 0.0052087
```

Decisao:

```text
saint_global_capped passou contra LoRA rank 2 em todos os regimes testados.
saint_global_capped ainda perdeu para full delta esparso com mesmo orcamento.
```

Conclusao atualizada:

```text
Fase 4 ainda permanece em andamento.
O gargalo agora nao e mais parameter_ratio contra LoRA;
o gargalo e competir contra budgeted_full_delta.
```

### Resultado de Escala + Residual

Foi adicionada a variante:

```text
saint_global_scaled_residual
```

Ela usa:

- score por ganho/custo;
- clustering k-means simples das assinaturas de gradiente;
- prototipos globais por cluster;
- escala treinavel por bloco com inicializacao por minimo quadrado;
- residual fino `2x2` selecionado depois de warmup;
- teto de parametros.

Resultado agregado:

```text
saint_global_scaled_residual test_loss medio: 0.0061497
saint_global_scaled_residual params medios: 106.7
saint_global_capped test_loss medio: 0.0074230
saint_global_capped params medios: 126.7
budgeted_full_delta_for_saint_global_scaled_residual test_loss medio: 0.0054662
```

Decisao:

```text
saint_global_scaled_residual passou contra LoRA rank 2 em todos os regimes.
saint_global_scaled_residual reduziu parametros contra saint_global_capped.
saint_global_scaled_residual melhorou qualidade media contra saint_global_capped.
saint_global_scaled_residual venceu budgeted_full_delta em 1 de 6 regimes.
saint_global_scaled_residual ainda perdeu para budgeted_full_delta na media.
```

Conclusao:

```text
Fase 4 continuou em andamento neste ponto.
O gargalo era melhorar qualidade por parametro, nao apenas reduzir parametros.
```

### Resultado do SAINT Dinamico

Foi adicionada a variante:

```text
saint_dynamic_delta
```

Ela usa:

- sensibilidade acumulada no warmup;
- residual escolhido por ganho marginal real por parametro;
- orcamento dinamico entre codebook, escala, bias e residual;
- bias treinavel por bloco;
- residual local `2x2`;
- residual low-rank local `4x4`;
- baseline `block_budgeted_delta`;
- LoRA tunado com ranks `1, 2, 4, 8`.

Resultado agregado:

```text
saint_dynamic_delta test_loss medio: 0.0055666
saint_dynamic_delta params medios: 111.0
lora_tuned_rank_2 test_loss medio: 0.0097656
lora_tuned_rank_4 test_loss medio: 0.0096352
block_budgeted_delta_for_saint_dynamic_delta test_loss medio: 0.0059000
budgeted_full_delta_for_saint_dynamic_delta test_loss medio: 0.0053768
```

Decisao:

```text
saint_dynamic_delta venceu LoRA rank 2 tunado em 6 de 6 regimes.
saint_dynamic_delta venceu LoRA rank 4 tunado em 6 de 6 regimes.
saint_dynamic_delta venceu block_budgeted_delta em 2 de 6 regimes.
saint_dynamic_delta venceu budgeted_full_delta em 2 de 6 regimes.
```

Criterio de fechamento aplicado:

```text
vence LoRA rank 2/4 ajustado em todos os regimes;
vence ou empata block_budgeted_delta em alguns regimes;
vence budgeted_full_delta em pelo menos 2 de 6 regimes;
mostra ganho claro contra LoRA em loss e ganho por parametro.
```

Resultado:

```text
Fase 4 concluida pelo criterio atual.
```

## 6. Fase 5 - Mini-Transformer

Status: **concluida**.

### Objetivo

Validar SAINT em um modelo com acoplamento real entre camadas.

### Implementacao Inicial

Foi criado um mini-transformer dependency-free em:

```text
saint/transformer/
```

Componentes:

- `model.py`: forward com embeddings, self-attention de ultima posicao, MLP e head;
- `training.py`: baselines `mini_full_delta`, `mini_budgeted_delta` e `mini_block_budgeted_delta`;
- `saint_adapter.py`: `mini_saint_dynamic_delta`;
- `benchmark.py`: sweep inicial;
- `scripts/benchmark_mini_transformer_phase5.py`;
- `tests/test_transformer_phase5.py`;
- `docs/process/fase_5_mini_transformer.md`.

O primeiro benchmark usa diferenca finita para medir gradientes contra loss
global. Isso e intencionalmente pequeno e auditavel; nao e o runtime final.

Resultado inicial:

```text
mini_saint_dynamic_delta test_loss medio: 0.00001998
mini_saint_dynamic_delta params medios: 48.0
mini_full_delta test_loss medio: 0.00002063
mini_full_delta params medios: 160.0
mini_budgeted_delta_for_saint test_loss medio: 0.00002063
mini_block_budgeted_delta_for_saint test_loss medio: 0.00002064
```

Leitura:

```text
SAINT venceu os controles no primeiro teste,
mas a tarefa ainda esta facil demais para concluir a fase.
```

Resultado com tarefa mais dificil:

```text
delta_scale: 3.0
mini_saint_dynamic_delta test_loss medio: 0.00020029
mini_saint_dynamic_delta params medios: 48.0
mini_saint_per_matrix_delta test_loss medio: 0.00019998
mini_saint_per_matrix_delta params medios: 72.0
mini_budgeted_delta_for_saint test_loss medio: 0.00020591
mini_block_budgeted_delta_for_saint test_loss medio: 0.00020598
mini_lora_rank_1 test_loss medio: 0.00020912
mini_lora_rank_2 test_loss medio: 0.00020912
```

Criterio automatico:

```text
SAINT global venceu LoRA rank 1 em 4/4 regimes.
SAINT global venceu LoRA rank 2 em 4/4 regimes.
SAINT global venceu budgeted_delta em 4/4 regimes.
SAINT global venceu block_budgeted_delta em 4/4 regimes.
SAINT global teve melhor eficiencia que SAINT por matriz em 4/4 regimes.
```

Resultado:

```text
Fase 5 concluida pelo criterio inicial.
```

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

Status: **concluida**.

### Objetivo

Escolher os blocos certos para treinar.

### Implementacao Inicial

Foi criado o modulo:

```text
saint/sensitivity/
```

Com benchmark:

```text
scripts/benchmark_sensitivity_phase6.py
```

E documentacao:

```text
docs/process/fase_6_mapa_sensibilidade.md
```

O primeiro experimento compara metodos de selecao com mesmo orcamento no
mini-transformer da Fase 5.

Resultado inicial:

```text
sensitivity_fisher test_loss medio: 0.00029264
sensitivity_gradient_norm test_loss medio: 0.00029264
sensitivity_gradient_weight test_loss medio: 0.00029269
sensitivity_random test_loss medio: 0.00029433
```

Decisao inicial:

```text
7 metodos venceram random.
criterio inicial passou.
Fase 6 continua em andamento ate testar blocos, regimes separados e uso no SAINT.
```

Resultado final:

```text
mini_saint_default_for_sensitivity test_loss medio: 0.00029131
mini_saint_gradient_norm test_loss medio: 0.00029131
sensitivity_accumulated_gradient test_loss medio: 0.00029264
sensitivity_fisher test_loss medio: 0.00029264
sensitivity_gradient_norm test_loss medio: 0.00029264
block_sensitivity_gradient_norm test_loss medio: 0.00029268
sensitivity_random test_loss medio: 0.00029433
```

Criterio final:

```text
repeated aprovado: sim
dense aprovado: sim
melhor bloco 2x2 venceu random: sim
sensibilidade acumulada venceu random: sim
SAINT alimentado por sensibilidade empatou/venceu SAINT padrao: sim
```

Resultado:

```text
Fase 6 concluida pelo criterio atual.
```

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

Status: **concluido em Python puro com testes unitarios**.

Segundo objetivo tecnico:

```text
detectar blocos iguais por assinatura simples
```

Status: **concluido para assinatura exata e quantizada**.

Terceiro objetivo tecnico:

```text
reportar erro de reconstrucao, reutilizacao e compressao estimada
```

Status: **concluido**.

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
