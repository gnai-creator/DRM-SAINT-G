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
Fase atual: Fase 15 - Escala 14B
Fase anterior: Fase 14 concluida com ressalvas
Proximo marco: Fase 15 Marco 1 - Ponte 14B Controlada
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
| 7 | Runtime SAINT | Concluida |
| 8 | Checkpoint e Reconstituicao | Concluida |
| 9 | Adaptador DRM Transformer | Concluida |
| 10 | Checkpoint Robusto | Concluida |
| 11 | Checkpoint Escalavel | Concluida |
| 12 | Validacao de Escala de Checkpoint | Concluida |
| 13 | Modelos Hugging Face Pequenos | Concluida com ressalvas |
| 14 | Escala 3B | Concluida com ressalvas |
| 15+ | Modelos reais e escala | Pendente |

## Fase 0 - Fundacao Conceitual

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

## Fase 1 - Biblioteca de Blocos

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

## Fase 2 - Benchmark de Reconstrucao

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

## Fase 3 - Roteador de Blocos

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

## Fase 4 - Treino de Camada Linear

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

## Fase 5 - Mini-Transformer

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

## Fase 6 - Mapa de Sensibilidade

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

## Fase 7 - Runtime SAINT

Status: **concluida**.

### Objetivo

Criar o runtime unificado.

### Implementacao Inicial

Foram adicionados:

```text
saint/config/
saint/memory/
saint/checkpoints/
saint/adapters/
saint/runtime/
saint/cli.py
configs/runtime_smoke.json
tests/test_runtime_phase7.py
docs/process/fase_7_runtime.md
```

O runtime inicial executa o mini-transformer com `mini_saint_dynamic_delta`,
mapa `gradient_norm`, config JSON, estimativa de memoria, logs e checkpoint.

Smoke test executado:

```text
python -m saint.cli inspect --config configs/runtime_smoke.json
python -m saint.cli estimate --config configs/runtime_smoke.json --vram-gb 12
python -m saint.cli train --config configs/runtime_smoke.json
python -m saint.cli resume --run runs/runtime_smoke
python -m saint.cli merge --run runs/runtime_smoke
```

Resultado:

```text
method: mini_saint_dynamic_delta
parameter_count: 30
test_loss: 0.00016531
fits_budget: true
artifacts: config.json, metrics.json, checkpoint.json, logs.jsonl, merged.json
```

Conclusao:

```text
Fase 7 concluida pelo criterio inicial.
```

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

## Fase 8 - Checkpoint e Reconstituicao

Status: **concluida**.

### Objetivo

Salvar deltas parciais e recompor o modelo final.

### Resultado

O runtime agora salva deltas reais para o mini-transformer:

```text
checkpoint.json:
  has_delta_payload: true
  delta_payload: {...}
```

E o merge reconstroi pesos mesclados:

```text
merged.json:
  merged: true
  merged_weights: {...}
```

Implementado em:

```text
saint/checkpoints/manager.py
saint/runtime/runner.py
saint/transformer/saint_adapter.py
saint/adapters/drm_transformer.py
pyproject.toml
docs/process/fase_8_checkpoint_reconstituicao.md
```

Conclusao:

```text
Fase 8 concluida para o mini-transformer dependency-free, com adapter inicial
de checkpoint do drm_transformer para inspecao e base de reconstituicao.
```

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

## Fase 9 - Adaptador DRM Transformer

Status: **concluida**.

### Objetivo

Usar `drm_transformer` como primeiro modelo customizado.

### Entregas

- adapter para carregar `DRMTransformer`;
- listagem de camadas e matrizes;
- congelar/descongelar partes;
- aplicar delta SAINT;
- salvar deltas;
- avaliar loss;
- trocar diferenca finita por autograd/PyTorch;
- medir gradientes reais para roteamento;
- comparar contra treino tradicional pequeno.

### Marco 1 - Integracao e Reconstituicao

Status: **concluido**.

O primeiro marco valida que o runtime SAINT consegue operar sobre pesos reais do
`drm_transformer`, mesmo antes de treinar com autograd.

Entregas:

- carregar `DRMTransformer` ou checkpoint do `drm_transformer`;
- listar matrizes treinaveis;
- mapear nomes de matriz para regioes SAINT;
- aplicar delta SAINT em pesos reais;
- salvar deltas em checkpoint SAINT;
- reconstituir pesos mesclados;
- validar shapes e compatibilidade de merge.

Resultado:

```text
drm_saint_delta_smoke:
  checkpoint -> matrizes 2D -> regioes SAINT -> delta_payload -> merged_weights
```

### Marco 2 - Treino Real com Autograd

Status: **concluido**.

O segundo marco troca o experimento dependency-free por treino PyTorch real.

Entregas:

- substituir diferenca finita por autograd;
- medir loss real do `drm_transformer`;
- medir gradientes reais por matriz/bloco;
- alimentar o roteador com sensibilidade por gradiente;
- comparar SAINT contra treino tradicional pequeno;
- gerar logs comparaveis com os experimentos anteriores.

Resultado:

```text
drm_saint_autograd_smoke:
  DRMTransformer pequeno -> loss real -> gradientes por bloco
  -> mascara SAINT por sensibilidade -> delta_payload -> merge
```

Smoke validado:

```text
initial_loss: 4.1506
saint_loss: 3.7953
full_baseline_loss: 3.7548
parameter_count: 32
shape_validation: true
```

### Criterio de conclusao

O `drm_transformer` deve treinar em modo SAINT em escala pequena e produzir logs comparaveis.

## Fase 10 - Checkpoint Robusto

Status: **concluida**.

### Objetivo

Transformar os checkpoints SAINT em artefatos compactos, verificaveis e
adequados para treinos reais.

Esta fase vem depois do adaptador `drm_transformer`, porque o formato robusto
deve refletir o que um treino com PyTorch/autograd realmente precisa salvar.

### Entregas

- deltas em formato binario/compacto;
- estado real do otimizador;
- checksums por arquivo e por payload;
- versao explicita do formato de checkpoint;
- validacao de integridade no `resume`;
- validacao de compatibilidade no `merge`;
- separacao entre metricas leves e payload pesado;
- suporte a checkpoints parciais por matriz/camada;
- documentacao do formato.

### Resultado

O runtime agora salva checkpoints robustos como manifesto leve mais payloads
compactos:

```text
checkpoint.json
deltas.saintbin
optimizer.saintopt
```

O `checkpoint.json` contem a versao do formato e checksums SHA-256 dos arquivos.
O `resume` valida os checksums, carrega estado de otimizador e rejeita payload
corrompido. O `merge` tambem valida integridade antes de reconstituir pesos.

Formato:

```text
format: saint_checkpoint
format_version: 1
```

Observacao: a Fase 12D atualizou o manifesto corrente para `format_version: 2`
e manteve migracao automatica de manifestos v1.

### Perguntas

- Qual e o menor formato suficiente para retomar treino?
- O checkpoint precisa salvar deltas materializados, codebooks, escalas e
  roteamento, ou todos eles?
- Quanto espaco o formato economiza contra JSON?
- A validacao detecta payload corrompido antes do merge?
- O formato continua legivel o bastante para debugging?

### Criterio de conclusao

SAINT deve conseguir:

```text
treinar -> salvar checkpoint compacto -> validar -> retomar -> fundir -> avaliar
```

com checksums e estado de otimizador preservados.

## Fase 11 - Checkpoint Escalavel

Status: **concluida**.

### Objetivo

Escalar o formato robusto para treinos maiores, com retomada real de otimizador,
payloads grandes e controle de compatibilidade entre versoes.

Esta fase aprofunda a Fase 10. A Fase 10 provou o formato robusto; a Fase 11
deve tornar esse formato apropriado para modelos maiores e runs longos.

### Entregas

- salvar estado completo de AdamW no caminho DRM autograd;
- adicionar suporte a shards grandes;
- adicionar mmap para payloads muito grandes;
- trocar `float32` por formatos opcionais como `float16`, `bfloat16` e quantizado;
- adicionar migracao entre versoes de checkpoint.

### Resultado

O checkpoint robusto agora suporta:

- estado real de AdamW no caminho `drm_saint_autograd_smoke`;
- retomada real de treino via `metadata.resume_run`;
- payloads de delta shardados;
- leitura de payload por `mmap`;
- dtypes `float32`, `float16`, `bfloat16` e `int8`;
- ponto de migracao de manifesto por `format_version`.

Smoke validado:

```text
first_loss: 4.1385
resume_initial_loss: 4.1385
second_loss: 4.1327
optimizer: AdamW
delta_format: saint_matrix_shards
dtype: float16
shards: 6
shape_validation: true
```

### Ordem Tecnica

1. Estado real de AdamW, porque sem isso `resume` ainda nao retoma treino de verdade.
2. Shards, porque modelos reais nao devem depender de um unico payload gigante.
3. Dtypes opcionais, porque reduzem tamanho e I/O.
4. `mmap`, para ler trechos sem carregar tudo na RAM.
5. Migracao entre versoes, quando houver mais de um formato real.

### Perguntas

- Qual parte do estado de AdamW precisa ser preservada por bloco?
- O shard deve ser por matriz, camada, tipo de tensor ou tamanho maximo?
- Qual dtype minimo mantem reconstrucao estavel?
- O merge consegue ler apenas os shards necessarios?
- Como validar compatibilidade entre formato antigo e novo?

### Criterio de conclusao

SAINT deve conseguir retomar um treino DRM autograd preservando estado real de
AdamW e usando checkpoints shardados/compactos com validacao de integridade.

## Fase 12 - Validacao de Escala de Checkpoint

Status: **concluida**.

### Objetivo

Validar o checkpoint escalavel em payloads maiores antes de entrar em modelos
Hugging Face pequenos.

A Fase 11 provou o formato em escala smoke. A Fase 12 deve medir se o formato
continua eficiente, integro e utilizavel quando o payload cresce.

### Subfases

#### Fase 12A - Validacao de Shards Grandes

Status: **concluida**.

- testar shards com checkpoints muito maiores;
- medir tempo de escrita;
- medir tempo de leitura;
- medir memoria usada no `resume`;
- medir memoria usada no `merge`;
- validar checksums por shard.

Resultado:

```text
format: saint_matrix_shards
matrix_count: 8
rows: 256
cols: 256
dtype: float16
shard_count: 16
payload_bytes: 1052000
write_elapsed_s: 0.1800
read_elapsed_s: 0.2984
read_peak_bytes: 17515952
checksum_validated: true
max_abs_error: 0.00000377
```

O formato agora tambem divide uma matriz individual grande em partes por faixa
de linhas e remonta a matriz original no carregamento.

#### Fase 12B - Merge Parcial

Status: **concluida**.

- fazer `merge` lendo apenas subconjuntos necessarios;
- permitir carregar apenas algumas matrizes;
- permitir carregar apenas algumas camadas;
- evitar materializar checkpoint inteiro quando o alvo for parcial;
- validar erro claro quando um shard necessario estiver ausente ou corrompido.

Resultado:

```text
matrix_count: 8
selected_count: 2
dtype: float16
shard_count: 16
full_read_elapsed_s: 0.2910
full_read_peak_bytes: 17524302
partial_read_elapsed_s: 0.0787
partial_read_peak_bytes: 4579606
max_abs_error: 0.00000189
```

O runtime agora aceita `merge_runtime(..., matrix_names={...})` e a CLI aceita
`saint merge --matrix <nome>` para fundir somente matrizes selecionadas.

#### Fase 12C - Custo de I/O por Dtype

Status: **concluida**.

- medir tamanho de checkpoint por dtype;
- medir tempo de escrita por dtype;
- medir tempo de leitura por dtype;
- comparar `float32`, `float16`, `bfloat16` e `int8`;
- registrar perda de precisao por dtype.

Resultado:

| dtype | bytes | razao vs float32 | shards | escrita s | leitura s | erro max abs |
|---|---:|---:|---:|---:|---:|---:|
| float32 | 2103968 | 1.0000 | 32 | 0.3124 | 0.3038 | 0.0000000005 |
| float16 | 1052000 | 0.5000 | 16 | 0.1771 | 0.3044 | 0.0000037720 |
| bfloat16 | 1052016 | 0.5000 | 16 | 0.1598 | 0.8330 | 0.0000602539 |
| int8 | 526121 | 0.2501 | 8 | 0.2240 | 0.2609 | 0.0000480315 |

`float16` e o melhor formato inicial para checkpoint compacto. `int8` reduz
mais tamanho, mas fica pendente de validacao de qualidade em tarefa real.

#### Fase 12D - Compatibilidade e Migracao

Status: **concluida**.

- adicionar uma migracao real quando `format_version` passar de 1 para 2;
- testar leitura de manifesto antigo;
- testar erro para versao futura incompativel;
- documentar campos estaveis e campos experimentais.

Resultado:

```text
manifesto atual: format_version 2
payload binario: payload_format_version 1
migracao: manifest_v1_to_v2
versao futura: rejeitada antes do resume/merge
```

O `checkpoint.json` agora possui campo `compatibility`, e manifestos v1 sao
migrados automaticamente para v2 com `migrated_from`.

#### Fase 12E - Qualidade Numerica

Status: **concluida**.

- validar `bfloat16` e `int8` contra perda de qualidade em tarefa real;
- comparar loss apos `resume` por dtype;
- comparar loss apos `merge` por dtype;
- definir quando cada dtype e aceitavel.

Resultado:

| dtype | bytes | merged_loss | delta loss vs float32 |
|---|---:|---:|---:|
| float32 | 1401 | 0.000123203766 | 0.000000000000 |
| float16 | 944 | 0.000123203746 | -0.000000000020 |
| bfloat16 | 945 | 0.000123204227 | 0.000000000461 |
| int8 | 799 | 0.000123203819 | 0.000000000053 |

`float16` fica como formato compacto inicial recomendado. `int8` passou no
mini-transformer pequeno, mas continua experimental para modelos reais maiores.

### Perguntas

- Qual tamanho de shard reduz I/O sem fragmentar demais o checkpoint?
- O `merge` parcial economiza RAM de forma mensuravel?
- Qual dtype tem melhor troca entre tamanho, velocidade e perda de qualidade?
- A migracao de formato detecta incompatibilidade antes de corromper o merge?
- `int8` e util para treino/resume ou apenas para distribuicao/merge final?

### Criterio de Conclusao

SAINT deve conseguir:

```text
checkpoint grande -> validar -> retomar -> merge parcial -> avaliar
```

com uso de memoria menor que carregar o payload completo e com perda numerica
medida por dtype.

### Resultado Final

```text
Fase 12 concluida.
```

A validacao de checkpoint agora cobre shards grandes, merge parcial, custo de
I/O por dtype, migracao de manifesto e qualidade numerica inicial por dtype.

## Fase 13 - Modelos Hugging Face Pequenos

Status: **concluida com ressalvas**.

### Objetivo

Testar SAINT em modelos reais pequenos.

### Marco 1 - Adaptador Local Dependency-Optional

Status: **concluido**.

Entregas:

- adapter `huggingface_causal_lm`;
- leitura de state dict JSON local;
- leitura opcional de `.bin`, `.pt` e `.pth` via PyTorch;
- tentativa opcional de `AutoModelForCausalLM.from_pretrained` com
  `local_files_only=True`;
- listagem de matrizes 2D por keywords;
- metodo `hf_saint_delta_smoke`;
- checkpoint robusto com dtype/shards;
- `inspect -> train -> resume -> merge`;
- config exemplo `configs/huggingface_smoke.json`;
- testes automatizados sem rede.

Limite:

```text
este marco valida integracao local com pesos Hugging Face,
mas ainda nao mede perplexity real nem executa autograd em transformers.
```

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

### Marco 2 - Treino Real com Autograd

Status: **concluido**.

Entregas:

- metodo `hf_saint_autograd_smoke`;
- modulo `saint/adapters/huggingface_autograd.py`;
- deltas treinaveis com PyTorch autograd;
- selecao de parametros por magnitude;
- otimizador AdamW;
- medicao de `initial_loss` e `train_loss`;
- exportacao de `delta_payload`;
- checkpoint robusto com dtype/shards;
- config exemplo `configs/huggingface_autograd_smoke.json`;
- teste que executa o fluxo completo quando PyTorch existe;
- erro claro quando PyTorch nao esta instalado.

Observacao:

```text
ambiente atual: torch 2.11.0+cu128, transformers 5.8.1, CUDA RTX 4090
```

### Marco 3 - Forward Real Transformers

Status: **concluido**.

Entregas:

- metodo `hf_saint_forward_smoke`;
- modulo `saint/adapters/huggingface_forward.py`;
- carregamento local com `AutoModelForCausalLM`;
- carregamento local com `AutoTokenizer`;
- tokenizacao de textos curtos;
- forward real com `model(input_ids, labels=input_ids)`;
- aplicacao de deltas por `torch.func.functional_call`;
- treino SAINT por autograd em matrizes alvo;
- medicao de loss inicial, loss final e perplexity;
- checkpoint robusto com dtype/shards;
- merge dos deltas treinados;
- config exemplo `configs/huggingface_forward_smoke.json`;
- teste com GPT-2 minimo local criado sem rede.

Fluxo validado:

```text
modelo local -> tokenizer local -> forward real -> treino SAINT -> checkpoint -> merge
```

### Marco 4 - Comparacao com Baselines HF

Status: **concluido**.

Entregas:

- modulo `saint/adapters/huggingface_benchmark.py`;
- benchmark `benchmark_hf_saint_vs_full`;
- comparacao SAINT vs full fine-tuning pequeno;
- seeds `31` e `32`;
- medicao de `tokens_per_s`;
- medicao de `cuda_peak_bytes`;
- checkpoint e merge avaliavel para SAINT.

Resultado CUDA:

| metodo | seed | parametros | loss inicial | loss final | delta loss | tokens/s | pico CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAINT | 31 | 8 | 2.792639 | 2.792619 | -0.000021 | 393.51 | 18230784 |
| full | 31 | 3824 | 2.790193 | 2.749064 | -0.041129 | 2915.51 | 18239488 |
| SAINT | 32 | 8 | 2.792639 | 2.792619 | -0.000021 | 5873.83 | 18230784 |
| full | 32 | 3824 | 2.767291 | 2.769696 | 0.002405 | 4872.50 | 18239488 |

### Marco 5 - LoRA e Modelo HF Real Local

Status: **concluido**.

Entregas:

- baseline `hf_lora_rank_2` no forward real;
- aplicacao LoRA por `torch.func.functional_call`, sem dependencia de `peft`;
- dataset curto ampliado;
- benchmark SAINT vs LoRA vs full fine-tuning;
- medicao de qualidade apos `resume`;
- medicao de ganho por parametro treinavel;
- suporte ao mesmo caminho local para modelos HF reais pequenos.

Metricas:

```text
resume_train_loss
resume_quality_delta
gain_per_parameter
tokens_per_s
cuda_peak_bytes
```

Observacao:

```text
o teste automatizado usa GPT-2 minimo local sem rede;
um modelo HF real local pode ser testado apontando model_path para o diretorio
do checkpoint ja existente na maquina.
```

Smoke CUDA:

```text
SAINT: params 8, loss 3.425533 -> 3.425443, ganho/param 0.00001124
LoRA r2: params 192, loss 3.432518 -> 3.432486, ganho/param 0.00000017
full: params 4064, loss 3.435688 -> 3.386605, ganho/param 0.00001208
resume_quality_delta SAINT: 0.0
```

### Marco 6 - Sweep HF Real Local

Status: **concluido**.

Entregas:

- modulo `saint/adapters/huggingface_sweep.py`;
- script `scripts/benchmark_huggingface_phase13.py`;
- sweep SAINT com budgets `4`, `8` e `16`;
- sweep LoRA com ranks `1`, `2`, `4` e `8`;
- resultado salvo em `results.json` e `results.md`;
- perplexity inicial, final e apos merge;
- memoria CUDA em run de 6 steps;
- execucao em `sshleifer/tiny-gpt2` salvo localmente.

Resultado CUDA:

| metodo | budget | rank | params | loss final | ppl merge | ganho/param | pico CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAINT | 4 |  | 4 | 10.824375 | 49923.772119 | 0.00000119 | 31205888 |
| SAINT | 8 |  | 8 | 10.824288 | 49919.201671 | 0.00001144 | 31205888 |
| SAINT | 16 |  | 12 | 10.824239 | 49916.583373 | 0.00001176 | 31205888 |
| LoRA |  | 1 | 12 | 10.818256 | 49923.962564 | 0.00000008 | 43878400 |
| LoRA |  | 2 | 24 | 10.818254 | 49923.867341 | 0.00000012 | 43878400 |
| LoRA |  | 4 | 48 | 10.818251 | 49923.676897 | 0.00000014 | 43878400 |
| LoRA |  | 8 | 96 | 10.818245 | 49923.391233 | 0.00000013 | 43878400 |
| full |  |  | 102714 | 10.806647 | 49347.742578 | 0.00000012 | 44737024 |

Leitura:

```text
SAINT ainda perde em loss absoluta contra full fine-tuning,
mas supera LoRA em ganho por parametro neste sweep curto e usa menos pico CUDA.
```

### Marco 7 - Dataset Externo e Validacao

Status: **concluido**.

Entregas:

- corpus `data/phase13_tiny_corpus.txt`;
- modulo `saint/adapters/huggingface_validation.py`;
- script `scripts/benchmark_huggingface_validation_phase13.py`;
- split treino/validacao;
- acumulacao de gradiente com `batch_size`;
- learning rates separados para SAINT, LoRA e full fine-tuning;
- checkpoint SAINT e artefato LoRA salvos;
- geracao curta antes e depois do merge SAINT;
- resultados em JSON/Markdown.

Resultado CUDA:

| metodo | budget | rank | params | val loss | ppl merge | artefato bytes | ganho/param |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAINT | 8 |  | 8 | 10.825989 | 50311.490508 | 27679 | 0.00000894 |
| LoRA |  | 4 | 48 | 10.825988 | 50311.442528 | 2757 | 0.00000230 |
| full |  |  | 102714 | 10.823018 | 50162.252171 | 0 | 0.00000004 |

Geracao curta:

```text
prompt: SAINT
base: SAINT stairs stairs stairs stairs stairs stairs stairs stairs
saint_merged: SAINT stairs stairs stairs stairs stairs stairs stairs stairs
```

Leitura:

```text
SAINT ainda perde em loss absoluta para full fine-tuning,
mas manteve melhor ganho por parametro que LoRA rank 4 neste run.
O formato SAINT atual salva mais contexto que o artefato LoRA,
entao ainda falta uma comparacao delta-only de tamanho.
```

### Marco 8 - Grid de Hiperparametros HF

Status: **concluido**.

Entregas:

- modulo `saint/adapters/huggingface_grid.py`;
- script `scripts/benchmark_huggingface_grid_phase13.py`;
- grid de budgets e learning rates SAINT;
- grid de ranks e learning rates LoRA;
- artefato SAINT delta-only;
- comparacao contra validation loss do modelo base;
- sanity check de geracao em multiplos prompts;
- resultados em JSON/Markdown.

Resultado CUDA:

| metodo | budget | rank | lr | params | val loss | delta vs base | ganho/param | bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAINT | 8 |  | 0.001 | 8 | 10.826005 | -0.000041 | 0.00000608 | 360 |
| SAINT | 8 |  | 0.005 | 8 | 10.825827 | -0.000219 | 0.00005305 | 348 |
| SAINT | 16 |  | 0.001 | 12 | 10.825991 | -0.000055 | 0.00000572 | 483 |
| SAINT | 16 |  | 0.005 | 12 | 10.825816 | -0.000230 | 0.00005500 | 460 |
| LoRA |  | 2 | 0.001 | 24 | 10.826046 | 0.000000 | 0.00000012 | 2733 |
| LoRA |  | 2 | 0.005 | 24 | 10.826030 | -0.000016 | 0.00000131 | 2733 |
| LoRA |  | 4 | 0.001 | 48 | 10.826044 | -0.000002 | 0.00000014 | 2797 |
| LoRA |  | 4 | 0.005 | 48 | 10.826013 | -0.000033 | 0.00000129 | 2797 |

Leitura:

```text
SAINT venceu LoRA neste grid curto em validation loss,
ganho por parametro e tamanho de artefato delta-only.
A geracao curta ainda nao mudou de forma observavel.
```

### Marco 9 - Dataset Real e Multiseed

Status: **concluido**.

Entregas:

- dataset externo pequeno `tinyshakespeare`;
- cache local em `data/tinyshakespeare_phase13.txt`;
- modulo `saint/adapters/huggingface_lora.py`;
- modulo `saint/adapters/huggingface_multiseed.py`;
- script `scripts/benchmark_huggingface_multiseed_phase13.py`;
- grid com seeds `31`, `32` e `33`;
- LoRA salvo, carregado e reaplicado no forward;
- avaliacao de geracao com prompts e metricas simples;
- decisao automatica de fechamento.

Resultado CUDA:

| metodo | count | mean val loss | best val loss | mean gain/param |
|---|---:|---:|---:|---:|
| SAINT | 12 | 10.823841 | 10.823558 | 0.00005602 |
| LoRA | 12 | 10.824103 | 10.824080 | 0.00000049 |

Artefato LoRA carregado:

```text
lora_loaded_validation_loss: 10.824079513549805
lora_loaded_perplexity: 50215.52463511919
```

Decisao:

```text
fase_13_can_close_with_caveat
```

Conclusao:

```text
Fase 13 fecha como prova de pipeline Hugging Face pequeno,
comparacao inicial contra LoRA e validacao multiseed.
A ressalva e que tiny-gpt2 e pequeno demais para provar qualidade de geracao.
```

### Proximo Marco

Fase 14 Marco 1 - Ponte HF Maior que Tiny GPT-2:

- escolher um modelo causal LM maior que `sshleifer/tiny-gpt2` e menor que 1B;
- rodar o mesmo benchmark multiseed;
- medir VRAM real;
- manter LoRA carregavel como controle;
- decidir se avanca para experimento 3B ou se precisa otimizar SAINT primeiro.

## Fase 14 - Escala 3B

Status: **em andamento**.

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

### Marco 1 - Ponte GPT-2 Small

Status: **concluido**.

Antes de 3B, foi usado `gpt2` como ponte maior que `sshleifer/tiny-gpt2`.

Modelo:

```text
gpt2
parametros: 124.439.808
```

Resultado CUDA:

| metodo | count | mean val loss | best val loss | mean gain/param |
|---|---:|---:|---:|---:|
| SAINT | 3 | 6.814889 | 6.814889 | 0.00000200 |
| LoRA | 3 | 6.808302 | 6.806123 | 0.00000399 |

Pico CUDA:

```text
SAINT: 2.083841536 GB
LoRA:  1.016675840 GB
```

Veredito:

```text
nao avancar ainda para 3B.
```

Motivo:

```text
LoRA venceu em loss, ganho por parametro e memoria no caminho atual.
SAINT precisa otimizar memoria/payload e melhorar selecao antes da escala 3B.
```

### Marco 2 - Otimizar SAINT em GPT-2 Small

Status: **concluido**.

Entregas:

- evitar segunda carga completa do modelo no adapter Hugging Face;
- salvar apenas deltas treinaveis no payload SAINT;
- reduzir custo de merge/eval;
- medir memoria por etapa;
- testar budgets SAINT maiores contra LoRA ranks `2` e `4`;
- decidir novamente se o projeto pode ir para 3B.

Resultado:

```text
SAINT: mean val loss 6.814783, best 6.814630, mean gain/param 0.00000276
LoRA:  mean val loss 6.803654, best 6.794116, mean gain/param 0.00000408
```

Curva CUDA:

| metodo | config | count | mean val loss | best val loss | mean gain/param | mean CUDA GB |
|---|---:|---:|---:|---:|---:|---:|
| SAINT | budget 16 | 3 | 6.814889 | 6.814889 | 0.00000200 | 2.079 |
| SAINT | budget 64 | 3 | 6.814830 | 6.814830 | 0.00000328 | 2.076 |
| SAINT | budget 256 | 3 | 6.814630 | 6.814630 | 0.00000302 | 2.076 |
| LoRA | rank 2 | 3 | 6.808302 | 6.806123 | 0.00000399 | 1.017 |
| LoRA | rank 4 | 3 | 6.799007 | 6.794116 | 0.00000418 | 1.017 |

Memoria por etapa em um run SAINT:

```text
load_cuda_peak_bytes: 508782592
train_cuda_peak_bytes: 2045830144
checkpoint_file_bytes: 273852
merge_cuda_peak_bytes: 18087936
```

Veredito:

```text
nao avancar ainda para 3B.
```

O caminho melhorou: nao ha segunda carga completa no treino, o payload SAINT
ficou esparso, e merge/eval usam matrizes selecionadas. Mesmo assim, LoRA ainda
vence em loss, ganho por parametro e pico CUDA.

### Marco 3 - Melhorar SAINT em GPT-2 Small

Status: **concluido**.

Objetivo:

```text
tornar SAINT competitivo contra LoRA rank 2/4 em GPT-2 small antes de 3B.
```

Entregas:

- selecionar deltas por gradiente real, nao apenas por magnitude inicial;
- testar mais matrizes alvo por camada;
- aumentar steps e medir se SAINT ganha mais com treino longo;
- comparar budgets maiores sem voltar a payload denso;
- reduzir overhead CUDA do forward funcional;
- manter LoRA rank `2` e `4` como controles obrigatorios.

Resultado:

```text
SAINT: mean val loss 6.776390, best 6.704383, mean gain/param 0.00033130
LoRA:  mean val loss 6.756175, best 6.727021, mean gain/param 0.00002527
```

Curva CUDA:

| metodo | config | count | mean val loss | best val loss | mean gain/param | mean CUDA GB |
|---|---:|---:|---:|---:|---:|---:|
| SAINT | budget 256 | 3 | 6.833445 | 6.833445 | 0.00068272 | 2.263 |
| SAINT | budget 1024 | 3 | 6.791341 | 6.791341 | 0.00022535 | 2.262 |
| SAINT | budget 4096 | 3 | 6.704383 | 6.704383 | 0.00008584 | 2.262 |
| LoRA | rank 2 | 3 | 6.771237 | 6.755111 | 0.00003135 | 1.018 |
| LoRA | rank 4 | 3 | 6.741114 | 6.727021 | 0.00001918 | 1.018 |

Veredito:

```text
SAINT ficou competitivo em qualidade, mas ainda nao em memoria.
```

O melhor SAINT venceu o melhor LoRA em validation loss, mas o pico CUDA
continuou aproximadamente 2.2x maior que LoRA.

Teste adicional com `--saint-lrs 0.005` e `--lora-lrs 0.005`:

```text
SAINT: mean val loss 6.562497, best 6.140045, mean gain/param 0.00049000
LoRA:  mean val loss 6.664005, best 6.563403, mean gain/param 0.00004904
```

Decisao automatica:

```text
fase_13_can_close_with_caveat
```

### Marco 4 - Reduzir Overhead CUDA do SAINT

Status: **concluido com ressalvas**.

Objetivo:

```text
reduzir o pico CUDA do caminho SAINT antes de tentar 3B.
```

Entregas:

- evitar `functional_call` com dicionario completo a cada step, se possivel;
- testar aplicacao temporaria dos deltas diretamente nos parametros alvo;
- medir o custo isolado do mapa de gradiente;
- reduzir matrizes carregadas no payload base para apenas alvos treinaveis;
- repetir GPT-2 small com `budget=4096` e steps maiores;
- decidir se a RTX 4090 tem margem suficiente para ponte 3B.

Resultado:

```text
SAINT: mean val loss 6.562497, best 6.140045, mean gain/param 0.00049000
LoRA:  mean val loss 6.664005, best 6.563403, mean gain/param 0.00004904
```

Memoria por etapa em um run SAINT:

```text
load_cuda_peak_bytes: 508782592
routing_cuda_peak_bytes: 2263763456
train_cuda_peak_bytes: 633309184
checkpoint_file_bytes: 19342
merge_cuda_peak_bytes: 18087936
```

Veredito:

```text
SAINT venceu em qualidade e reduziu artifact/checkpoint de benchmark,
mas o pico CUDA ainda e dominado pelo roteamento por gradiente.
```

### Marco 5 - Reduzir Memoria do Roteamento

Status: **concluido com ressalvas**.

Objetivo:

```text
calcular sensibilidade por gradiente sem materializar custo alto de todas as
matrizes alvo ao mesmo tempo.
```

Entregas:

- calcular gradiente uma matriz alvo por vez;
- mover scores para CPU antes do top-k global;
- limpar cache CUDA entre matrizes;
- comparar roteamento completo contra roteamento aproximado barato;
- repetir GPT-2 small com `budget=4096`, `lr=0.005`;
- decidir se a ponte 3B pode iniciar.

Resultado:

| roteamento | SAINT mean val loss | SAINT best val loss | routing CUDA GB | train CUDA GB | decisao |
|---|---:|---:|---:|---:|---|
| gradient completo | 6.562497 | 6.140045 | 2.264 | 0.638 | passa com ressalva |
| gradient sequencial | 6.140045 | 6.140045 | 2.247 | 0.637 | passa com ressalva |
| magnitude | 6.751935 | 6.751935 | 0.518 | 0.637 | falha contra LoRA |

Controle LoRA:

```text
LoRA: mean val loss 6.664005, best 6.563403, mean gain/param 0.00004904
```

Veredito:

```text
O sequencial preserva qualidade, mas nao reduz memoria o bastante.
Magnitude mostra o piso barato, mas perde qualidade.
```

### Marco 6 - Roteamento Aproximado de Baixo Custo

Status: **concluido**.

Objetivo:

```text
aproximar o beneficio do gradiente sem rodar backward completo caro.
```

Entregas:

- testar gradiente de ultima camada ou `lm_head` como proxy;
- testar sensibilidade por ativacao sem backward completo;
- testar score hibrido `magnitude * ativacao`;
- testar subset de batch/seq_len menor apenas para roteamento;
- comparar qualidade/memoria contra `gradient_sequential`;
- decidir se GPT-2 small autoriza ponte 3B com ressalva.

Resultado:

| roteamento | SAINT mean val loss | SAINT best val loss | routing CUDA GB | train CUDA GB | decisao |
|---|---:|---:|---:|---:|---|
| gradient sequencial completo | 6.140045 | 6.140045 | 2.259 | 0.637 | passa com ressalva |
| gradient sequencial subset | 6.441031 | 6.441031 | 0.540 | 0.637 | passa com ressalva |
| activation subset | 6.522398 | 6.522398 | 0.519 | 0.613 | passa com ressalva |
| magnitude activation subset | 6.716236 | 6.716236 | 0.528 | 0.637 | falha contra LoRA |
| lm_head proxy subset | 6.716236 | 6.716236 | 0.528 | 0.637 | falha contra LoRA |

Controle LoRA:

```text
LoRA: mean val loss 6.664005, best 6.563403, mean gain/param 0.00004904
```

Veredito:

```text
GPT-2 small autoriza uma ponte 3B experimental com ressalva.
```

Condicoes:

- usar `activation` como roteador inicial;
- manter `gradient_sequential` como controle de qualidade;
- micro-batch 1;
- `routing_max_length` baixo;
- checkpoint somente com deltas esparsos;
- abortar se pico CUDA passar do budget definido.

### Marco 7 - Ponte 3B Controlada

Status: **concluido com ressalva**.

Objetivo:

```text
testar um modelo proximo de 3B sem prometer treino completo.
```

Entregas:

- modelo escolhido: `Qwen/Qwen2.5-3B`;
- carga CUDA validada com dtype economico;
- smoke de load/forward sem treino passou;
- SAINT `activation` rodou com `budget=4096`, micro-batch 1 e
  `routing_max_length=8`;
- checkpoint salvou delta esparso real com 4096 valores;
- resume e merge foram validados;
- LoRA rank 1 coube como controle minimo;
- load/routing/train/checkpoint/merge foram medidos.

Resultado SAINT 3B:

| metrica | valor |
|---|---:|
| base validation loss | 7.690704 |
| SAINT validation loss | 7.688824 |
| merged validation loss | 7.688824 |
| parametros treinaveis | 4096 |
| delta sparse values | 4096 |
| checkpoint bytes | 309252 |
| load CUDA GB | 5.863 |
| routing CUDA GB | 5.864 |
| train CUDA GB | 6.016 |
| peak CUDA GB | 11.869 |
| tokens/s | 97.155 |

Comparacao minima:

| metodo | val loss | params | gain/param | CUDA peak GB |
|---|---:|---:|---:|---:|
| SAINT activation | 7.688824 | 4096 | 0.00000613 | 11.869 |
| LoRA rank 1 | 7.664804 | 6400 | 0.00000684 | 7.630 |

Veredito:

```text
Fase 14 Marco 7 passou tecnicamente, mas Fase 14 ainda nao fecha.
```

Motivo:

- SAINT ja carrega, treina, salva, retoma e mergeia em 3B;
- LoRA rank 1 ainda vence em qualidade e pico CUDA neste microteste;
- o proximo marco precisa reduzir overhead e melhorar ganho por parametro.

### Marco 8 - Otimizacao SAINT 3B contra LoRA

Status: **concluido com ressalva**.

Mudancas:

- treino HF deixou de materializar recortes densos de pesos para checkpoint;
- `saint_sparse_delta` salva coordenadas reais e shapes completos;
- validacao de checkpoint esparso nao expande payload para matriz densa;
- benchmarks HF aplicam deltas por coordenada;
- `bfloat16` passou a ser o dtype operacional para 3B.

Grid activation:

| metodo | count | mean val loss | best val loss | mean gain/param |
|---|---:|---:|---:|---:|
| SAINT activation | 4 | 7.657179 | 7.656769 | 0.00000225 |
| LoRA rank 1/2 | 4 | 7.671440 | 7.661643 | 0.00000430 |

Controle `gradient_sequential` subset:

| metodo | budget | val loss | params | routing CUDA GB | train CUDA GB | merge CUDA GB |
|---|---:|---:|---:|---:|---:|---:|
| gradient_sequential | 8192 | 7.460260 | 8192 | 5.956 | 5.969 | 8.084 |
| gradient_sequential | 16384 | 7.448014 | 16384 | 5.956 | 5.970 | 8.084 |

Veredito:

```text
Fase 14 Marco 8 passou, mas Fase 14 ainda nao fecha.
```

Motivo:

- SAINT activation venceu LoRA em validation loss media e melhor loss;
- `gradient_sequential` subset foi muito melhor que activation e LoRA;
- o pico do forward funcional SAINT ainda fica maior que LoRA.

### Marco 9 - Reducao do Pico Funcional

Status: **concluido**.

Mudancas:

- adicionado `--saint-delta-application`;
- `functional` preserva o caminho de qualidade com `functional_call`;
- `inplace` aplica delta no peso real, faz rollback e atualiza coordenadas
  esparsas pelo gradiente do peso alvo;
- merge/eval agora separa pico de load e pico de forward com delta aplicado.

Comparacao:

| delta application | val loss | train loss | peak CUDA GB | train CUDA GB | merge load GB | merge eval GB |
|---|---:|---:|---:|---:|---:|---:|
| functional | 7.448014 | 7.389300 | 11.785 | 5.970 | 5.870 | 8.084 |
| inplace | 7.687493 | 7.981713 | 7.660 | 5.970 | 5.870 | 8.059 |

Repeticao multiseed do baseline:

| seed | val loss | train loss | peak CUDA GB | routing CUDA GB | train CUDA GB |
|---:|---:|---:|---:|---:|---:|
| 31 | 7.448014 | 7.389300 | 11.785 | 5.956 | 5.970 |
| 32 | 7.448014 | 7.389300 | 11.785 | 5.956 | 5.970 |
| 33 | 7.448014 | 7.389300 | 11.785 | 5.956 | 5.970 |

Decisao:

```text
Fase 14 conclui com ressalva.
```

Baseline para a proxima fase:

```text
Qwen/Qwen2.5-3B
gradient_sequential subset
budget 16384
delta_application functional
bfloat16
micro-batch 1
```

Ressalva:

- LoRA ainda tem menor pico CUDA;
- o caminho `inplace` reduz memoria, mas ainda nao preserva qualidade;
- Fase 15 deve introduzir offload e reduzir materializacao densa antes de 14B.

## Fase 15 - Escala 14B

Status: **em andamento**.

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

### Marco 1 - Ponte 14B Controlada

Status: **concluido com bloqueio de treino**.

Modelo:

```text
Qwen/Qwen2.5-14B
```

Mudancas:

- adicionada dependencia `accelerate>=1.12`;
- adicionados argumentos de offload HF:
  - `--hf-device-map`;
  - `--hf-max-memory`;
  - `--hf-offload-folder`;
- criado `scripts/benchmark_huggingface_phase15_14b.py`;
- load HF centralizado em `saint.adapters.huggingface_loading`.

Smoke:

| metrica | valor |
|---|---:|
| checkpoint local | 29.55 GB |
| load_s | 26.964 |
| forward_s | 1.653 |
| loss | 7.261498 |
| load CUDA GB | 18.634 |
| forward CUDA GB | 19.834 |

Device map:

```text
GPU: embeddings + camadas 0 a 32
CPU: camadas 33 a 47 + norm + rotary_emb + lm_head
```

Tentativa SAINT:

```text
inplace + activation + budget 4096 + micro-batch 1
timeout apos 20 minutos
```

Veredito:

```text
14B load/forward e viavel com offload CPU.
14B treino ainda esta bloqueado por latencia.
```

Proximo marco:

- limitar treino a uma unica matriz em camada residente na GPU;
- evitar tocar camadas offloadadas durante treino;
- testar delta somente em `model.layers.0.self_attn.q_proj.weight`;
- medir custo com diferentes `max_memory`;
- so comparar LoRA 14B depois de SAINT completar um step abaixo de 5 minutos.

## Fase 16 - Escala 70B

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

## Fase 17 - Otimizacoes

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

## Fase 18 - Avaliacao

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

## Fase 19 - Produto de Pesquisa

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
