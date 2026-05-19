# Roadmap DRM-SAINT-G

DRM-SAINT-G significa **DRM por Enxerto com DRM-SAINT-G-Phi**. Este roadmap organiza o desenvolvimento do paradigma em fases verificaveis, partindo de experimentos matematicos pequenos ate testes em modelos grandes.

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

O principio do DRM-SAINT-G e:

```text
loss global,
atualizacao local,
padroes compartilhados,
calculo agrupado,
recomposicao final
```

## Status Atual

```text
Fase atual: Fase DRM-G - DRM-SAINT-G
Fase anterior: Fase 15 concluida com ressalvas
Proximo marco: DRM-G Marco 3 - Consolidacao permanente do enxerto
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
| 7 | Runtime DRM-SAINT-G | Concluida |
| 8 | Checkpoint e Reconstituicao | Concluida |
| 9 | Adaptador DRM Transformer | Concluida |
| 10 | Checkpoint Robusto | Concluida |
| 11 | Checkpoint Escalavel | Concluida |
| 12 | Validacao de Escala de Checkpoint | Concluida |
| 13 | Modelos Hugging Face Pequenos | Concluida com ressalvas |
| 14 | Escala 3B | Concluida com ressalvas |
| 15 | Escala 14B | Concluida com ressalvas |
| DRM-G | DRM-SAINT-G | Em andamento |
| 16+ | Modelos reais e escala maior | Pendente |

## Fase 0 - Fundacao Conceitual

Status: **concluida**.

### Objetivo

Consolidar o paradigma antes de escrever um runtime grande.

### Entregas

- `docs/paradigma_treino_tradicional.md`
- `docs/paradigma_DRM-SAINT-G.md`
- `docs/arquitetura.md`
- `docs/roadmap.md`
- `docs/glossario_tecnico.md`
- `docs/hipoteses.md`
- `docs/criterios_sucesso_falha.md`

### Perguntas

- O que DRM-SAINT-G tenta melhorar em relacao ao treino tradicional?
- O que DRM-SAINT-G nao promete?
- Quais baselines serao usados?
- Quais experimentos invalidam a ideia?

### Criterio de conclusao

A fase termina quando o projeto tem uma definicao clara:

```text
DRM-SAINT-G = sparse multi-scale block-codebook delta training
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

- pacote inicial `drm-saint-g` - concluido;
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

A fase termina quando DRM-SAINT-G consegue:

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
bruto, mas e coerente com a proposta de DRM-SAINT-G para treino de deltas: congelar
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

DRM-SAINT-G deve empatar ou superar pelo menos uma baseline eficiente em algum eixo relevante:

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
DRM-SAINT-G_routed_delta test_loss: 0.0004701, params: 56
lora_rank_2 test_loss: 0.0034705, params: 32
sparse_sensitivity_delta test_loss: 0.0010289, params: 16
```

Leitura:

```text
DRM-SAINT-G ainda nao bate full_delta em loss,
mas aprende com menos parametros e supera LoRA rank 2 nesse caso sintetico.
```

A Fase 4 continua em andamento porque falta sweep de sementes, ranks de LoRA,
orcamentos do roteador e criterio automatico de sucesso/falha.

### Resultado do Sweep

Foi adicionado sweep com 5 sementes, LoRA rank 1/2/4 e tres orcamentos DRM-SAINT-G.

Melhor variante DRM-SAINT-G:

```text
DRM-SAINT-G_routed_f50_c25
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

Resultado inicial:

```text
DRM-SAINT-G_routed_f50_c25 passou contra lora_rank_2.
DRM-SAINT-G_routed_f25_c50 passou contra lora_rank_2.
DRM-SAINT-G_routed_f25_c25 falhou contra lora_rank_1 por exceder 2x parametros.
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
DRM-SAINT-G_routed_f50_c25 test_loss medio: 0.0043511
DRM-SAINT-G_routed_f50_c25 params medios: 364.0
lora_rank_2 test_loss medio: 0.0098293
lora_rank_2 params medios: 74.7
budgeted_full_delta_for_DRM-SAINT-G_routed_f50_c25 test_loss medio: 0.0035874
```

Leitura:

```text
DRM-SAINT-G ainda vence LoRA rank 2 em loss,
mas perde o criterio de parametros em 16x16 e 32x32.
Contra full_delta esparso com orcamento equivalente, DRM-SAINT-G ainda perde.
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
DRM-SAINT-G_global_capped
```

Ela usa codebook global por camada, limite de regioes livres e limite global de
prototipos.

Resultado agregado:

```text
DRM-SAINT-G_global_capped test_loss medio: 0.0074230
DRM-SAINT-G_global_capped params medios: 126.7
lora_rank_2 test_loss medio: 0.0098293
lora_rank_2 params medios: 74.7
budgeted_full_delta_for_DRM-SAINT-G_global_capped test_loss medio: 0.0052087
```

Decisao:

```text
DRM-SAINT-G_global_capped passou contra LoRA rank 2 em todos os regimes testados.
DRM-SAINT-G_global_capped ainda perdeu para full delta esparso com mesmo orcamento.
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
DRM-SAINT-G_global_scaled_residual
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
DRM-SAINT-G_global_scaled_residual test_loss medio: 0.0061497
DRM-SAINT-G_global_scaled_residual params medios: 106.7
DRM-SAINT-G_global_capped test_loss medio: 0.0074230
DRM-SAINT-G_global_capped params medios: 126.7
budgeted_full_delta_for_DRM-SAINT-G_global_scaled_residual test_loss medio: 0.0054662
```

Decisao:

```text
DRM-SAINT-G_global_scaled_residual passou contra LoRA rank 2 em todos os regimes.
DRM-SAINT-G_global_scaled_residual reduziu parametros contra DRM-SAINT-G_global_capped.
DRM-SAINT-G_global_scaled_residual melhorou qualidade media contra DRM-SAINT-G_global_capped.
DRM-SAINT-G_global_scaled_residual venceu budgeted_full_delta em 1 de 6 regimes.
DRM-SAINT-G_global_scaled_residual ainda perdeu para budgeted_full_delta na media.
```

Conclusao:

```text
Fase 4 continuou em andamento neste ponto.
O gargalo era melhorar qualidade por parametro, nao apenas reduzir parametros.
```

### Resultado do DRM-SAINT-G Dinamico

Foi adicionada a variante:

```text
DRM-SAINT-G_dynamic_delta
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
DRM-SAINT-G_dynamic_delta test_loss medio: 0.0055666
DRM-SAINT-G_dynamic_delta params medios: 111.0
lora_tuned_rank_2 test_loss medio: 0.0097656
lora_tuned_rank_4 test_loss medio: 0.0096352
block_budgeted_delta_for_DRM-SAINT-G_dynamic_delta test_loss medio: 0.0059000
budgeted_full_delta_for_DRM-SAINT-G_dynamic_delta test_loss medio: 0.0053768
```

Decisao:

```text
DRM-SAINT-G_dynamic_delta venceu LoRA rank 2 tunado em 6 de 6 regimes.
DRM-SAINT-G_dynamic_delta venceu LoRA rank 4 tunado em 6 de 6 regimes.
DRM-SAINT-G_dynamic_delta venceu block_budgeted_delta em 2 de 6 regimes.
DRM-SAINT-G_dynamic_delta venceu budgeted_full_delta em 2 de 6 regimes.
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

Validar DRM-SAINT-G em um modelo com acoplamento real entre camadas.

### Implementacao Inicial

Foi criado um mini-transformer dependency-free em:

```text
saint/transformer/
```

Componentes:

- `model.py`: forward com embeddings, self-attention de ultima posicao, MLP e head;
- `training.py`: baselines `mini_full_delta`, `mini_budgeted_delta` e `mini_block_budgeted_delta`;
- `DRM-SAINT-G_adapter.py`: `mini_DRM-SAINT-G_dynamic_delta`;
- `benchmark.py`: sweep inicial;
- `scripts/benchmark_mini_transformer_phase5.py`;
- `tests/test_transformer_phase5.py`;
- `docs/process/fase_5_mini_transformer.md`.

O primeiro benchmark usa diferenca finita para medir gradientes contra loss
global. Isso e intencionalmente pequeno e auditavel; nao e o runtime final.

Resultado inicial:

```text
mini_DRM-SAINT-G_dynamic_delta test_loss medio: 0.00001998
mini_DRM-SAINT-G_dynamic_delta params medios: 48.0
mini_full_delta test_loss medio: 0.00002063
mini_full_delta params medios: 160.0
mini_budgeted_delta_for_DRM-SAINT-G test_loss medio: 0.00002063
mini_block_budgeted_delta_for_DRM-SAINT-G test_loss medio: 0.00002064
```

Leitura:

```text
DRM-SAINT-G venceu os controles no primeiro teste,
mas a tarefa ainda esta facil demais para concluir a fase.
```

Resultado com tarefa mais dificil:

```text
delta_scale: 3.0
mini_DRM-SAINT-G_dynamic_delta test_loss medio: 0.00020029
mini_DRM-SAINT-G_dynamic_delta params medios: 48.0
mini_DRM-SAINT-G_per_matrix_delta test_loss medio: 0.00019998
mini_DRM-SAINT-G_per_matrix_delta params medios: 72.0
mini_budgeted_delta_for_DRM-SAINT-G test_loss medio: 0.00020591
mini_block_budgeted_delta_for_DRM-SAINT-G test_loss medio: 0.00020598
mini_lora_rank_1 test_loss medio: 0.00020912
mini_lora_rank_2 test_loss medio: 0.00020912
```

Criterio automatico:

```text
DRM-SAINT-G global venceu LoRA rank 1 em 4/4 regimes.
DRM-SAINT-G global venceu LoRA rank 2 em 4/4 regimes.
DRM-SAINT-G global venceu budgeted_delta em 4/4 regimes.
DRM-SAINT-G global venceu block_budgeted_delta em 4/4 regimes.
DRM-SAINT-G global teve melhor eficiencia que DRM-SAINT-G por matriz em 4/4 regimes.
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
- DRM-SAINT-G por camada;
- DRM-SAINT-G por matriz;
- DRM-SAINT-G por blocos;
- DRM-SAINT-G com codebook multi-escala;
- DRM-SAINT-G com consolidacao.

### Perguntas

- Loss global com atualizacao local funciona?
- Treino por partes converge?
- Consolidacao reduz conflito?
- A ordem de treino importa muito?
- O mapa de sensibilidade escolhe boas partes?

### Criterio de conclusao

Prosseguir se DRM-SAINT-G aprender de forma consistente e nao apenas memorizar comportamento acidental.

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
Fase 6 continua em andamento ate testar blocos, regimes separados e uso no DRM-SAINT-G.
```

Resultado final:

```text
mini_DRM-SAINT-G_default_for_sensitivity test_loss medio: 0.00029131
mini_DRM-SAINT-G_gradient_norm test_loss medio: 0.00029131
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
DRM-SAINT-G alimentado por sensibilidade empatou/venceu DRM-SAINT-G padrao: sim
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

## Fase 7 - Runtime DRM-SAINT-G

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

O runtime inicial executa o mini-transformer com `mini_DRM-SAINT-G_dynamic_delta`,
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
method: mini_DRM-SAINT-G_dynamic_delta
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

- CLI `drm-saint-g`;
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
drm-saint-g inspect --model ./model
drm-saint-g reconstruct --matrix ./weights.pt
drm-saint-g estimate --model ./model --vram-gb 12
drm-saint-g train --config configs/exp.yaml
drm-saint-g resume --run runs/exp001
drm-saint-g merge --run runs/exp001
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
saint/transformer/DRM-SAINT-G_adapter.py
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

DRM-SAINT-G deve conseguir:

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
- aplicar delta DRM-SAINT-G;
- salvar deltas;
- avaliar loss;
- trocar diferenca finita por autograd/PyTorch;
- medir gradientes reais para roteamento;
- comparar contra treino tradicional pequeno.

### Marco 1 - Integracao e Reconstituicao

Status: **concluido**.

O primeiro marco valida que o runtime DRM-SAINT-G consegue operar sobre pesos reais do
`drm_transformer`, mesmo antes de treinar com autograd.

Entregas:

- carregar `DRMTransformer` ou checkpoint do `drm_transformer`;
- listar matrizes treinaveis;
- mapear nomes de matriz para regioes DRM-SAINT-G;
- aplicar delta DRM-SAINT-G em pesos reais;
- salvar deltas em checkpoint DRM-SAINT-G;
- reconstituir pesos mesclados;
- validar shapes e compatibilidade de merge.

Resultado:

```text
drm_DRM-SAINT-G_delta_smoke:
  checkpoint -> matrizes 2D -> regioes DRM-SAINT-G -> delta_payload -> merged_weights
```

### Marco 2 - Treino Real com Autograd

Status: **concluido**.

O segundo marco troca o experimento dependency-free por treino PyTorch real.

Entregas:

- substituir diferenca finita por autograd;
- medir loss real do `drm_transformer`;
- medir gradientes reais por matriz/bloco;
- alimentar o roteador com sensibilidade por gradiente;
- comparar DRM-SAINT-G contra treino tradicional pequeno;
- gerar logs comparaveis com os experimentos anteriores.

Resultado:

```text
drm_DRM-SAINT-G_autograd_smoke:
  DRMTransformer pequeno -> loss real -> gradientes por bloco
  -> mascara DRM-SAINT-G por sensibilidade -> delta_payload -> merge
```

Smoke validado:

```text
initial_loss: 4.1506
DRM-SAINT-G_loss: 3.7953
full_baseline_loss: 3.7548
parameter_count: 32
shape_validation: true
```

### Criterio de conclusao

O `drm_transformer` deve treinar em modo DRM-SAINT-G em escala pequena e produzir logs comparaveis.

## Fase 10 - Checkpoint Robusto

Status: **concluida**.

### Objetivo

Transformar os checkpoints DRM-SAINT-G em artefatos compactos, verificaveis e
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
deltas.DRM-SAINT-Gbin
optimizer.DRM-SAINT-Gopt
```

O `checkpoint.json` contem a versao do formato e checksums SHA-256 dos arquivos.
O `resume` valida os checksums, carrega estado de otimizador e rejeita payload
corrompido. O `merge` tambem valida integridade antes de reconstituir pesos.

Formato:

```text
format: DRM-SAINT-G_checkpoint
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

DRM-SAINT-G deve conseguir:

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

- estado real de AdamW no caminho `drm_DRM-SAINT-G_autograd_smoke`;
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
delta_format: DRM-SAINT-G_matrix_shards
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

DRM-SAINT-G deve conseguir retomar um treino DRM autograd preservando estado real de
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
format: DRM-SAINT-G_matrix_shards
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
`drm-saint-g merge --matrix <nome>` para fundir somente matrizes selecionadas.

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

DRM-SAINT-G deve conseguir:

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

Testar DRM-SAINT-G em modelos reais pequenos.

### Marco 1 - Adaptador Local Dependency-Optional

Status: **concluido**.

Entregas:

- adapter `huggingface_causal_lm`;
- leitura de state dict JSON local;
- leitura opcional de `.bin`, `.pt` e `.pth` via PyTorch;
- tentativa opcional de `AutoModelForCausalLM.from_pretrained` com
  `local_files_only=True`;
- listagem de matrizes 2D por keywords;
- metodo `hf_DRM-SAINT-G_delta_smoke`;
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

DRM-SAINT-G deve mostrar vantagem ou comportamento complementar a LoRA em pelo menos um tipo de tarefa.

### Marco 2 - Treino Real com Autograd

Status: **concluido**.

Entregas:

- metodo `hf_DRM-SAINT-G_autograd_smoke`;
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

- metodo `hf_DRM-SAINT-G_forward_smoke`;
- modulo `saint/adapters/huggingface_forward.py`;
- carregamento local com `AutoModelForCausalLM`;
- carregamento local com `AutoTokenizer`;
- tokenizacao de textos curtos;
- forward real com `model(input_ids, labels=input_ids)`;
- aplicacao de deltas por `torch.func.functional_call`;
- treino DRM-SAINT-G por autograd em matrizes alvo;
- medicao de loss inicial, loss final e perplexity;
- checkpoint robusto com dtype/shards;
- merge dos deltas treinados;
- config exemplo `configs/huggingface_forward_smoke.json`;
- teste com GPT-2 minimo local criado sem rede.

Fluxo validado:

```text
modelo local -> tokenizer local -> forward real -> treino DRM-SAINT-G -> checkpoint -> merge
```

### Marco 4 - Comparacao com Baselines HF

Status: **concluido**.

Entregas:

- modulo `saint/adapters/huggingface_benchmark.py`;
- benchmark `benchmark_hf_DRM-SAINT-G_vs_full`;
- comparacao DRM-SAINT-G vs full fine-tuning pequeno;
- seeds `31` e `32`;
- medicao de `tokens_per_s`;
- medicao de `cuda_peak_bytes`;
- checkpoint e merge avaliavel para DRM-SAINT-G.

Resultado CUDA:

| metodo | seed | parametros | loss inicial | loss final | delta loss | tokens/s | pico CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| DRM-SAINT-G | 31 | 8 | 2.792639 | 2.792619 | -0.000021 | 393.51 | 18230784 |
| full | 31 | 3824 | 2.790193 | 2.749064 | -0.041129 | 2915.51 | 18239488 |
| DRM-SAINT-G | 32 | 8 | 2.792639 | 2.792619 | -0.000021 | 5873.83 | 18230784 |
| full | 32 | 3824 | 2.767291 | 2.769696 | 0.002405 | 4872.50 | 18239488 |

### Marco 5 - LoRA e Modelo HF Real Local

Status: **concluido**.

Entregas:

- baseline `hf_lora_rank_2` no forward real;
- aplicacao LoRA por `torch.func.functional_call`, sem dependencia de `peft`;
- dataset curto ampliado;
- benchmark DRM-SAINT-G vs LoRA vs full fine-tuning;
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
DRM-SAINT-G: params 8, loss 3.425533 -> 3.425443, ganho/param 0.00001124
LoRA r2: params 192, loss 3.432518 -> 3.432486, ganho/param 0.00000017
full: params 4064, loss 3.435688 -> 3.386605, ganho/param 0.00001208
resume_quality_delta DRM-SAINT-G: 0.0
```

### Marco 6 - Sweep HF Real Local

Status: **concluido**.

Entregas:

- modulo `saint/adapters/huggingface_sweep.py`;
- script `scripts/benchmark_huggingface_phase13.py`;
- sweep DRM-SAINT-G com budgets `4`, `8` e `16`;
- sweep LoRA com ranks `1`, `2`, `4` e `8`;
- resultado salvo em `results.json` e `results.md`;
- perplexity inicial, final e apos merge;
- memoria CUDA em run de 6 steps;
- execucao em `sshleifer/tiny-gpt2` salvo localmente.

Resultado CUDA:

| metodo | budget | rank | params | loss final | ppl merge | ganho/param | pico CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| DRM-SAINT-G | 4 |  | 4 | 10.824375 | 49923.772119 | 0.00000119 | 31205888 |
| DRM-SAINT-G | 8 |  | 8 | 10.824288 | 49919.201671 | 0.00001144 | 31205888 |
| DRM-SAINT-G | 16 |  | 12 | 10.824239 | 49916.583373 | 0.00001176 | 31205888 |
| LoRA |  | 1 | 12 | 10.818256 | 49923.962564 | 0.00000008 | 43878400 |
| LoRA |  | 2 | 24 | 10.818254 | 49923.867341 | 0.00000012 | 43878400 |
| LoRA |  | 4 | 48 | 10.818251 | 49923.676897 | 0.00000014 | 43878400 |
| LoRA |  | 8 | 96 | 10.818245 | 49923.391233 | 0.00000013 | 43878400 |
| full |  |  | 102714 | 10.806647 | 49347.742578 | 0.00000012 | 44737024 |

Leitura:

```text
DRM-SAINT-G ainda perde em loss absoluta contra full fine-tuning,
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
- learning rates separados para DRM-SAINT-G, LoRA e full fine-tuning;
- checkpoint DRM-SAINT-G e artefato LoRA salvos;
- geracao curta antes e depois do merge DRM-SAINT-G;
- resultados em JSON/Markdown.

Resultado CUDA:

| metodo | budget | rank | params | val loss | ppl merge | artefato bytes | ganho/param |
|---|---:|---:|---:|---:|---:|---:|---:|
| DRM-SAINT-G | 8 |  | 8 | 10.825989 | 50311.490508 | 27679 | 0.00000894 |
| LoRA |  | 4 | 48 | 10.825988 | 50311.442528 | 2757 | 0.00000230 |
| full |  |  | 102714 | 10.823018 | 50162.252171 | 0 | 0.00000004 |

Geracao curta:

```text
prompt: DRM-SAINT-G
base: DRM-SAINT-G stairs stairs stairs stairs stairs stairs stairs stairs
DRM-SAINT-G_merged: DRM-SAINT-G stairs stairs stairs stairs stairs stairs stairs stairs
```

Leitura:

```text
DRM-SAINT-G ainda perde em loss absoluta para full fine-tuning,
mas manteve melhor ganho por parametro que LoRA rank 4 neste run.
O formato DRM-SAINT-G atual salva mais contexto que o artefato LoRA,
entao ainda falta uma comparacao delta-only de tamanho.
```

### Marco 8 - Grid de Hiperparametros HF

Status: **concluido**.

Entregas:

- modulo `saint/adapters/huggingface_grid.py`;
- script `scripts/benchmark_huggingface_grid_phase13.py`;
- grid de budgets e learning rates DRM-SAINT-G;
- grid de ranks e learning rates LoRA;
- artefato DRM-SAINT-G delta-only;
- comparacao contra validation loss do modelo base;
- sanity check de geracao em multiplos prompts;
- resultados em JSON/Markdown.

Resultado CUDA:

| metodo | budget | rank | lr | params | val loss | delta vs base | ganho/param | bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DRM-SAINT-G | 8 |  | 0.001 | 8 | 10.826005 | -0.000041 | 0.00000608 | 360 |
| DRM-SAINT-G | 8 |  | 0.005 | 8 | 10.825827 | -0.000219 | 0.00005305 | 348 |
| DRM-SAINT-G | 16 |  | 0.001 | 12 | 10.825991 | -0.000055 | 0.00000572 | 483 |
| DRM-SAINT-G | 16 |  | 0.005 | 12 | 10.825816 | -0.000230 | 0.00005500 | 460 |
| LoRA |  | 2 | 0.001 | 24 | 10.826046 | 0.000000 | 0.00000012 | 2733 |
| LoRA |  | 2 | 0.005 | 24 | 10.826030 | -0.000016 | 0.00000131 | 2733 |
| LoRA |  | 4 | 0.001 | 48 | 10.826044 | -0.000002 | 0.00000014 | 2797 |
| LoRA |  | 4 | 0.005 | 48 | 10.826013 | -0.000033 | 0.00000129 | 2797 |

Leitura:

```text
DRM-SAINT-G venceu LoRA neste grid curto em validation loss,
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
| DRM-SAINT-G | 12 | 10.823841 | 10.823558 | 0.00005602 |
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
- decidir se avanca para experimento 3B ou se precisa otimizar DRM-SAINT-G primeiro.

## Fase 14 - Escala 3B

Status: **concluida com ressalvas**.

### Objetivo

Primeiro teste serio em GPU domestica.

### Configuracao Esperada

```text
modelo: 3B
VRAM alvo: 12GB ou 24GB
modo: base congelada + deltas DRM-SAINT-G
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
| DRM-SAINT-G | 3 | 6.814889 | 6.814889 | 0.00000200 |
| LoRA | 3 | 6.808302 | 6.806123 | 0.00000399 |

Pico CUDA:

```text
DRM-SAINT-G: 2.083841536 GB
LoRA:  1.016675840 GB
```

Veredito:

```text
nao avancar ainda para 3B.
```

Motivo:

```text
LoRA venceu em loss, ganho por parametro e memoria no caminho atual.
DRM-SAINT-G precisa otimizar memoria/payload e melhorar selecao antes da escala 3B.
```

### Marco 2 - Otimizar DRM-SAINT-G em GPT-2 Small

Status: **concluido**.

Entregas:

- evitar segunda carga completa do modelo no adapter Hugging Face;
- salvar apenas deltas treinaveis no payload DRM-SAINT-G;
- reduzir custo de merge/eval;
- medir memoria por etapa;
- testar budgets DRM-SAINT-G maiores contra LoRA ranks `2` e `4`;
- decidir novamente se o projeto pode ir para 3B.

Resultado:

```text
DRM-SAINT-G: mean val loss 6.814783, best 6.814630, mean gain/param 0.00000276
LoRA:  mean val loss 6.803654, best 6.794116, mean gain/param 0.00000408
```

Curva CUDA:

| metodo | config | count | mean val loss | best val loss | mean gain/param | mean CUDA GB |
|---|---:|---:|---:|---:|---:|---:|
| DRM-SAINT-G | budget 16 | 3 | 6.814889 | 6.814889 | 0.00000200 | 2.079 |
| DRM-SAINT-G | budget 64 | 3 | 6.814830 | 6.814830 | 0.00000328 | 2.076 |
| DRM-SAINT-G | budget 256 | 3 | 6.814630 | 6.814630 | 0.00000302 | 2.076 |
| LoRA | rank 2 | 3 | 6.808302 | 6.806123 | 0.00000399 | 1.017 |
| LoRA | rank 4 | 3 | 6.799007 | 6.794116 | 0.00000418 | 1.017 |

Memoria por etapa em um run DRM-SAINT-G:

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

O caminho melhorou: nao ha segunda carga completa no treino, o payload DRM-SAINT-G
ficou esparso, e merge/eval usam matrizes selecionadas. Mesmo assim, LoRA ainda
vence em loss, ganho por parametro e pico CUDA.

### Marco 3 - Melhorar DRM-SAINT-G em GPT-2 Small

Status: **concluido**.

Objetivo:

```text
tornar DRM-SAINT-G competitivo contra LoRA rank 2/4 em GPT-2 small antes de 3B.
```

Entregas:

- selecionar deltas por gradiente real, nao apenas por magnitude inicial;
- testar mais matrizes alvo por camada;
- aumentar steps e medir se DRM-SAINT-G ganha mais com treino longo;
- comparar budgets maiores sem voltar a payload denso;
- reduzir overhead CUDA do forward funcional;
- manter LoRA rank `2` e `4` como controles obrigatorios.

Resultado:

```text
DRM-SAINT-G: mean val loss 6.776390, best 6.704383, mean gain/param 0.00033130
LoRA:  mean val loss 6.756175, best 6.727021, mean gain/param 0.00002527
```

Curva CUDA:

| metodo | config | count | mean val loss | best val loss | mean gain/param | mean CUDA GB |
|---|---:|---:|---:|---:|---:|---:|
| DRM-SAINT-G | budget 256 | 3 | 6.833445 | 6.833445 | 0.00068272 | 2.263 |
| DRM-SAINT-G | budget 1024 | 3 | 6.791341 | 6.791341 | 0.00022535 | 2.262 |
| DRM-SAINT-G | budget 4096 | 3 | 6.704383 | 6.704383 | 0.00008584 | 2.262 |
| LoRA | rank 2 | 3 | 6.771237 | 6.755111 | 0.00003135 | 1.018 |
| LoRA | rank 4 | 3 | 6.741114 | 6.727021 | 0.00001918 | 1.018 |

Veredito:

```text
DRM-SAINT-G ficou competitivo em qualidade, mas ainda nao em memoria.
```

O melhor DRM-SAINT-G venceu o melhor LoRA em validation loss, mas o pico CUDA
continuou aproximadamente 2.2x maior que LoRA.

Teste adicional com `--saint-lrs 0.005` e `--lora-lrs 0.005`:

```text
DRM-SAINT-G: mean val loss 6.562497, best 6.140045, mean gain/param 0.00049000
LoRA:  mean val loss 6.664005, best 6.563403, mean gain/param 0.00004904
```

Decisao automatica:

```text
fase_13_can_close_with_caveat
```

### Marco 4 - Reduzir Overhead CUDA do DRM-SAINT-G

Status: **concluido com ressalvas**.

Objetivo:

```text
reduzir o pico CUDA do caminho DRM-SAINT-G antes de tentar 3B.
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
DRM-SAINT-G: mean val loss 6.562497, best 6.140045, mean gain/param 0.00049000
LoRA:  mean val loss 6.664005, best 6.563403, mean gain/param 0.00004904
```

Memoria por etapa em um run DRM-SAINT-G:

```text
load_cuda_peak_bytes: 508782592
routing_cuda_peak_bytes: 2263763456
train_cuda_peak_bytes: 633309184
checkpoint_file_bytes: 19342
merge_cuda_peak_bytes: 18087936
```

Veredito:

```text
DRM-SAINT-G venceu em qualidade e reduziu artifact/checkpoint de benchmark,
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

| roteamento | DRM-SAINT-G mean val loss | DRM-SAINT-G best val loss | routing CUDA GB | train CUDA GB | decisao |
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

| roteamento | DRM-SAINT-G mean val loss | DRM-SAINT-G best val loss | routing CUDA GB | train CUDA GB | decisao |
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
- DRM-SAINT-G `activation` rodou com `budget=4096`, micro-batch 1 e
  `routing_max_length=8`;
- checkpoint salvou delta esparso real com 4096 valores;
- resume e merge foram validados;
- LoRA rank 1 coube como controle minimo;
- load/routing/train/checkpoint/merge foram medidos.

Resultado DRM-SAINT-G 3B:

| metrica | valor |
|---|---:|
| base validation loss | 7.690704 |
| DRM-SAINT-G validation loss | 7.688824 |
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
| DRM-SAINT-G activation | 7.688824 | 4096 | 0.00000613 | 11.869 |
| LoRA rank 1 | 7.664804 | 6400 | 0.00000684 | 7.630 |

Veredito:

```text
Fase 14 Marco 7 passou tecnicamente, mas Fase 14 ainda nao fecha.
```

Motivo:

- DRM-SAINT-G ja carrega, treina, salva, retoma e mergeia em 3B;
- LoRA rank 1 ainda vence em qualidade e pico CUDA neste microteste;
- o proximo marco precisa reduzir overhead e melhorar ganho por parametro.

### Marco 8 - Otimizacao DRM-SAINT-G 3B contra LoRA

Status: **concluido com ressalva**.

Mudancas:

- treino HF deixou de materializar recortes densos de pesos para checkpoint;
- `DRM-SAINT-G_sparse_delta` salva coordenadas reais e shapes completos;
- validacao de checkpoint esparso nao expande payload para matriz densa;
- benchmarks HF aplicam deltas por coordenada;
- `bfloat16` passou a ser o dtype operacional para 3B.

Grid activation:

| metodo | count | mean val loss | best val loss | mean gain/param |
|---|---:|---:|---:|---:|
| DRM-SAINT-G activation | 4 | 7.657179 | 7.656769 | 0.00000225 |
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

- DRM-SAINT-G activation venceu LoRA em validation loss media e melhor loss;
- `gradient_sequential` subset foi muito melhor que activation e LoRA;
- o pico do forward funcional DRM-SAINT-G ainda fica maior que LoRA.

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

Tentativa DRM-SAINT-G:

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

- reduzir pico de memoria antes de tentar qualidade;
- evitar segunda carga completa no caminho 14B;
- criar modo `train-only` sem base eval, merge eval e generation;
- medir load, routing, train e checkpoint separadamente.

### Marco 2 - Alvo Unico em Camada Residente

Status: **concluido como diagnostico, ainda sem treino viavel**.

Mudancas:

- `target_names` explicito para DRM-SAINT-G HF;
- filtro `target_device` para evitar matrizes offloadadas;
- CLI multiseed com:
  - `--saint-target-names`;
  - `--saint-target-device`;
- novo probe `scripts/benchmark_huggingface_phase15_target_probe.py`.

Probe:

| matriz | device | shape |
|---|---|---:|
| `model.layers.0.self_attn.q_proj.weight` | `cuda:0` | 5120x5120 |
| `model.layers.33.self_attn.q_proj.weight` | `meta`/CPU offload | 5120x5120 |

Smoke relativo:

| modelo/config | load_s | forward_s | forward CUDA GB |
|---|---:|---:|---:|
| Qwen2.5-3B sem offload | 15.242 | 0.339 | 5.874 |
| Qwen2.5-14B `0=18GiB,cpu=64GiB` | 33.053 | 1.353 | 17.783 |
| Qwen2.5-14B `0=22GiB,cpu=64GiB` | 23.044 | 1.098 | 21.885 |

Tentativa DRM-SAINT-G limitada:

```text
target: model.layers.0.self_attn.q_proj.weight
budget: 1024
steps: 1
resultado: CUDA budget exceeded during train: 29.856 GB
```

Veredito:

```text
selecionar uma matriz residente funciona;
o caminho de treino/validacao ainda gera pico de memoria alto demais.
```

Proximo marco:

- repetir train-only com budgets maiores;
- testar faixas de `max_memory`;
- adicionar avaliacao posterior opcional em processo separado;
- testar LoRA 14B rank 1 somente no mesmo limite de CUDA.

### Marco 3 - Train-Only 14B Sem Grid

Status: **concluido**.

Mudancas:

- criado `scripts/benchmark_huggingface_phase15_train_only.py`;
- adicionado modo `train_only` no caminho HF;
- removidos base eval, merge eval e generation do smoke 14B;
- ativado `gradient_checkpointing` opcional;
- registrados tempos de load, routing, train e checkpoint payload;
- checkpoint gerado pelo runtime sem recarregar modelo.

Resultado principal:

| metrica | valor |
|---|---:|
| modelo | Qwen2.5-14B |
| target | `model.layers.0.self_attn.q_proj.weight` |
| max_memory | `0=14GiB,cpu=64GiB` |
| status | ok |
| elapsed_s_total | 45.948 |
| load_s | 37.341 |
| routing_s | 2.219 |
| train_s | 3.444 |
| checkpoint_payload_s | 0.000463 |
| train CUDA GB | 17.401 |
| parametros treinaveis | 1024 |
| delta JSON bytes | 81818 |

Veredito:

```text
DRM-SAINT-G completou um step autograd 14B abaixo de 5 minutos com checkpoint.
```

Limite observado:

```text
0=18GiB,cpu=64GiB ainda estoura no treino: 29.647 GB.
```

Proximo marco:

- reduzir pico da avaliacao posterior abaixo de 23 GB;
- medir loss antes/depois sem duplicar memoria;
- repetir saint/LoRA com mais steps;
- testar targets `v_proj` e `o_proj`.

### Marco 4 - Comparacao Train-Only 14B

Status: **concluido com ressalva na avaliacao posterior**.

Mudancas:

- criado `scripts/benchmark_huggingface_phase15_compare.py`;
- cada ponto DRM-SAINT-G roda em subprocesso separado;
- criado `scripts/benchmark_huggingface_phase15_eval_checkpoint.py`;
- LoRA rank 1 testado no mesmo alvo;
- resultados salvos em JSON/Markdown.

Resultados DRM-SAINT-G:

| max_memory | budget | status | train_s | train CUDA GB |
|---|---:|---|---:|---:|
| `0=12GiB,cpu=64GiB` | 1024 | ok | 3.818 | 15.769 |
| `0=12GiB,cpu=64GiB` | 4096 | ok | 3.794 | 15.769 |
| `0=12GiB,cpu=64GiB` | 8192 | ok | 3.774 | 15.769 |
| `0=14GiB,cpu=64GiB` | 1024 | ok | 3.691 | 17.401 |
| `0=14GiB,cpu=64GiB` | 4096 | ok | 3.631 | 17.401 |
| `0=14GiB,cpu=64GiB` | 8192 | ok | 3.486 | 17.401 |
| `0=16GiB,cpu=64GiB` | 1024 | ok | 3.004 | 19.032 |
| `0=16GiB,cpu=64GiB` | 4096 | ok | 3.017 | 19.032 |
| `0=16GiB,cpu=64GiB` | 8192 | ok | 3.052 | 19.033 |

LoRA rank 1:

| metrica | valor |
|---|---:|
| status | ok |
| train_s | 4.729 |
| train CUDA GB | 17.453 |
| parametros treinaveis | 10240 |

Avaliacoes:

```text
drm-saint-g train-only passou abaixo de 23 GB em todos os pontos.
LoRA rank 1 tambem coube no limite.
Avaliacao posterior separada funciona, mas merge/eval chegou a 29.682 GB.
```

Proximo marco:

- repetir `v_proj` com seeds diferentes;
- testar camadas 0, 1 e 2;
- ajustar scheduler/LR para steps maiores;
- testar `v_proj + o_proj`;
- medir validacao com mais exemplos.

### Marco 5 - Qualidade e Avaliacao 14B

Status: **concluido**.

Mudancas:

- `train_only` mede `initial_loss`, `loss_delta` e `gain_per_parameter`;
- avaliacao posterior usa `torch.no_grad()`;
- LoRA rank 1 usa o mesmo texto do corpus;
- targets testados: `q_proj`, `v_proj`, `o_proj`.

Reducao do pico de avaliacao:

| checkpoint | validation loss | eval CUDA GB |
|---|---:|---:|
| Marco 4 `q_proj` budget 8192 | 6.801508 | 12.487 |
| Marco 5 `v_proj` budget 16384 | 5.612469 | 12.487 |

Qualidade train-only:

| target | budget | loss delta | ganho/param | train CUDA GB |
|---|---:|---:|---:|---:|
| `q_proj` | 8192 | -0.021804 | 2.6616e-06 | 15.821 |
| `q_proj` | 16384 | -0.012429 | 7.5862e-07 | 15.821 |
| `v_proj` | 8192 | -0.298339 | 3.6418e-05 | 15.779 |
| `v_proj` | 16384 | -0.393503 | 2.4017e-05 | 15.779 |
| `o_proj` | 8192 | -0.055394 | 6.7620e-06 | 15.821 |

Comparacao LoRA rank 1:

| metodo | params | loss delta | ganho/param |
|---|---:|---:|---:|
| DRM-SAINT-G `q_proj` budget 8192 | 8192 | -0.021804 | 2.6616e-06 |
| LoRA rank 1 `q_proj` | 10240 | 0.000000 | 0.0000 |

Veredito:

```text
Marco 5 passou: DRM-SAINT-G 14B reduziu loss e avaliacao posterior ficou abaixo de 23 GB.
```

### Marco 6 - Robustez de Qualidade 14B

Status: **concluido com ressalva de validacao**.

Mudancas:

- `lr_decay` no treino DRM-SAINT-G in-place;
- LoRA rank 1 com `B` nao-zero;
- avaliacao posterior com base e merged via `--include-base`;
- testados seeds 31/32/33, camadas 0/1/2, scheduler e `v_proj + o_proj`.

Seeds em `layer0.v_proj`, budget 8192:

| seed | DRM-SAINT-G loss delta | LoRA loss delta |
|---:|---:|---:|
| 31 | -0.298339 | +0.012675 |
| 32 | -0.298339 | -0.002483 |
| 33 | -0.298339 | -0.028358 |

Camadas em `v_proj`, seed 31:

| camada | DRM-SAINT-G loss delta | ganho/param |
|---:|---:|---:|
| 0 | -0.298339 | 3.6418e-05 |
| 1 | -0.542579 | 6.6233e-05 |
| 2 | -0.748944 | 9.1424e-05 |

Outros resultados:

| teste | DRM-SAINT-G loss delta | LoRA loss delta |
|---|---:|---:|
| `v_proj`, steps 8, `lr_decay=0.8` | -0.313544 | -0.000988 |
| `v_proj + o_proj`, layer 0 | -0.045656 | -0.003168 |

Validacao com 4 exemplos:

| metrica | valor |
|---|---:|
| base_validation_loss | 6.882289 |
| merged_validation_loss | 6.892014 |
| validation_loss_delta | +0.009724 |
| merge load CUDA GB | 20.261 |
| merge eval CUDA GB | 12.500 |

Veredito:

```text
DRM-SAINT-G ficou robusto em train loss, mas a validacao ainda nao melhorou.
```

Proximo marco:

- selecionar blocos por mini-validacao;
- usar mais de um texto de treino;
- medir validacao durante treino sem recarregar modelo;
- comparar LoRA rank 1/2 com o mesmo numero de textos;
- adicionar early stopping por validation loss.

### Marco 7 - Generalizacao 14B

Status: **concluido como diagnostico negativo**.

Mudancas:

- treino esparso in-place modularizado em
  `saint.adapters.huggingface_sparse_train`;
- `train_only` passou a aceitar multiplos textos de treino;
- validacao durante treino sem recarregar modelo;
- early stopping por validation loss;
- comparacao LoRA rank 1/2 com o mesmo numero de textos;
- roteamentos `validation_gradient` e `validation_magnitude_activation`.

Resultado:

| teste | metodo | status/loss delta | pico CUDA |
|---|---|---:|---:|
| layer 2 `v_proj` | `validation_gradient` | falhou | 29.962 GB |
| layer 2 `v_proj` com 8GiB | `validation_gradient` | falhou | 29.987 GB |
| layer 2 `v_proj` | DRM-SAINT-G validation proxy | +0.073896 | 15.782 GB |
| layer 2 `v_proj` | LoRA rank 1 | -0.003053 | 15.785 GB |
| layer 3 `v_proj` | DRM-SAINT-G validation proxy | +0.058203 | 15.782 GB |
| layer 3 `v_proj` | LoRA rank 1 | -0.000484 | 15.785 GB |
| layer 2 `v_proj` | DRM-SAINT-G activation multitexto | -0.182964 | 15.782 GB |
| layer 2/3 `v_proj` | LoRA rank 2 | falhou | 26.703 GB |

Veredito:

```text
validacao durante treino funciona, mas o roteamento por mini-validacao ainda
nao e competitivo no 14B.
```

Leitura:

- `validation_gradient` e fiel, mas estoura o limite de 23 GB;
- `validation_magnitude_activation` cabe, mas escolheu blocos piores que
  `activation`;
- `activation` ainda e o baseline DRM-SAINT-G 14B mais forte neste marco;
- LoRA rank 1 melhora pouco, enquanto LoRA rank 2 ainda estoura memoria no
  caminho atual.

Proximo marco:

- selecionar candidatos por `activation` e ranquear apenas um subconjunto por
  mini-validacao;
- medir ganho real de validacao por candidato sem backward completo quando
  possivel;
- aplicar rollback antes de escolher o delta final;
- tornar LoRA rank 2 esparso/low-rank sem update denso temporario;
- manter `activation` como baseline 14B ate o roteador de validacao vencer.

### Marco 8 - Roteamento de Validacao Barato

Status: **em andamento**.

Mudancas:

- criado `activation_validation_rerank`;
- candidatos sao selecionados por `activation`;
- um subconjunto e ranqueado por mini-validacao com deltas temporarios;
- cada probe usa rollback antes do proximo candidato;
- LoRA rank 1/2 passou a usar `forward_hook`, sem materializar `A @ B` como
  delta denso temporario.

Nota sobre LoRA:

```text
LoRA forward-hook baseline ainda e LoRA:
W' = W + A @ B

A diferenca e operacional:
y = x @ W.T + x @ B.T @ A.T
```

Isso evita criar uma matriz densa temporaria. Portanto, quando o LoRA rank 2
passa a caber, isso nao e uma versao adulterada do LoRA; e um baseline mais
justo e mais proximo do uso real.

Resultado parcial em Qwen2.5-14B, layer 2 `v_proj`, `train_texts=3`,
`validation_texts=6`:

| metodo | loss delta | params | ganho/param | CUDA treino |
|---|---:|---:|---:|---:|
| DRM-SAINT-G `activation_validation_rerank` | -0.090874 | 8192 | 1.1093e-05 | 15.782 GB |
| DRM-SAINT-G `activation` | -0.086226 | 8192 | 1.0526e-05 | 15.782 GB |
| LoRA rank 1 forward-hook | -0.132801 | 6144 | 2.1615e-05 | 15.778 GB |
| LoRA rank 2 forward-hook | -0.573527 | 12288 | 4.6674e-05 | 15.778 GB |

Veredito parcial:

```text
o rerank barato cabe em memoria e melhora um pouco contra activation,
mas ainda perde para LoRA forward-hook.
```

Proximo marco:

- fazer DRM-SAINT-G operar em blocos 2x2/4x4 no treino HF, nao apenas coordenadas
  independentes;
- usar mini-validacao para escolher blocos;
- comparar budgets menores;
- medir validation loss antes/depois no mesmo processo;
- testar camadas 1, 2 e 3;
- manter LoRA forward-hook rank 1/2 como baseline obrigatorio.

### Marco 9 - Blocos Treinaveis por Validacao

Status: **concluido como experimento inicial**.

Mudancas:

- criado `activation_block_validation_rerank`;
- adicionado `--routing-block-size`;
- blocos 2x2/4x4 sao escolhidos por ativacao e ranqueados por mini-validacao;
- cada bloco usa delta temporario com rollback;
- adicionado `--validation-rerank-max-candidates`;
- resultados registram `initial_validation_loss`, `validation_loss` e
  `validation_loss_delta`.

Resultado principal em Qwen2.5-14B, layer 2 `v_proj`:

| metodo | train delta | validation delta | params efetivos | routing_s | CUDA treino |
|---|---:|---:|---:|---:|---:|
| DRM-SAINT-G block4 rerank | -0.012179 | -0.013615 | 128 | 34.756 | 15.781 GB |
| LoRA rank 1 forward-hook | -0.227059 | n/a | 6144 | n/a | 15.778 GB |

Outros achados:

- `block_size=2` com 64 candidatos ficou caro e ruim:
  `routing_s=242.675`, `train_delta=+0.026165`;
- `block_size=4` usa melhor o budget, mas 64 candidatos ainda custam
  `routing_s=238.497`;
- limitar a 8 candidatos reduz o roteamento para cerca de 35s;
- DRM-SAINT-G block4 mostrou sinal de validacao positiva com poucos parametros, mas
  LoRA forward-hook continua mais forte.

Veredito:

```text
blocos treinaveis funcionam e a validacao real esta registrada,
mas a qualidade ainda nao compete com LoRA.
```

Proximo marco:

- usar blocos estruturados, por exemplo `delta = scale * prototype`;
- agrupar varios candidatos em um unico forward de validacao;
- reduzir custo de `routing_s`;
- medir validation loss corrigida em camadas 1, 2 e 3;
- comparar contra LoRA forward-hook em validation loss.

### Marco 10 - Blocos Estruturados

Status: **concluido como diagnostico**.

Mudancas:

- criado `activation_structured_block_validation_rerank`;
- bloco estruturado usa `delta_bloco = scale * prototype`;
- cada bloco tem um parametro treinavel `scale`;
- prototipo inicial usa sinal do peso base;
- criado `--validation-rerank-batch-size`;
- LoRA forward-hook passou a registrar validation loss.

Resultado 14B, `v_proj`, block4, budget 16:

| camada | DRM-SAINT-G val delta | DRM-SAINT-G params | routing_s | LoRA r1 val delta | LoRA r2 val delta |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.002546 | 16 | 16.888 | -0.116323 | -0.180719 |
| 2 | -0.002882 | 16 | 16.282 | -0.285410 | -0.197755 |
| 3 | +0.021051 | 16 | 16.304 | -0.255529 | -0.450788 |

Veredito:

```text
o rerank batelado reduziu routing_s de cerca de 35s para 16s,
mas bloco estruturado scale * prototype ainda nao compete com LoRA.
```

Proximo marco:

- usar dois ou mais prototipos por bloco;
- inicializar prototipos por gradiente/ativacao;
- testar budgets 32, 64 e 128 scales;
- comparar ganho por parametro em validation loss;
- decidir se DRM-SAINT-G 14B deve otimizar compressao extrema ou qualidade contra
  LoRA.

### Marco 11 - Multiplos Prototipos Estruturados

Status: **concluido como sinal de compressao extrema**.

Mudancas:

- blocos estruturados aceitam multiplos prototipos:
  `delta = scale_a * P_a + scale_b * P_b`;
- prototipos podem usar `weight_sign`, `activation`, `weight_activation` ou
  `weight_value`;
- scales podem ser por bloco, linha ou coluna;
- metadata registra `prototype_count`, `prototype_shape`, `scale_id_shape` e
  `scale_count`.

Resultado 14B, layer 2 `v_proj`, `prototype_count=2`,
`prototype_mode=weight_activation`, `scale_granularity=row`:

| budget | validation delta | params | ganho val/param | routing_s |
|---:|---:|---:|---:|---:|
| 32 | -0.021731 | 32 | 6.79e-04 | 28.497 |
| 64 | -0.007499 | 64 | 1.17e-04 | 48.258 |
| 128 | -0.003388 | 128 | 2.65e-05 | 92.410 |

Comparacao:

| metodo | validation delta | params | ganho val/param |
|---|---:|---:|---:|
| DRM-SAINT-G budget 32 | -0.021731 | 32 | 6.79e-04 |
| LoRA rank 1 forward-hook | -0.165617 | 6144 | 2.70e-05 |

Veredito:

```text
LoRA ainda vence em qualidade absoluta.
DRM-SAINT-G venceu em ganho de validacao por parametro no budget 32.
```

Decisao tecnica:

```text
Fase 15 deve tratar DRM-SAINT-G como compressao extrema primeiro,
nao como substituto direto de LoRA em qualidade absoluta.
```

Proximo marco:

- reportar `validation_gain_per_parameter` automaticamente;
- testar budgets 16/32/64 em mais camadas;
- comparar contra baseline com parametro equivalente;
- varrer `prototype_mode` e `scale_granularity`;
- fechar criterio da Fase 15 por eficiencia por parametro/checkpoint.

### Marco 12 - Eficiencia por Parametro

Status: **concluido**.

Mudancas:

- criado `scripts/benchmark_huggingface_phase15_efficiency.py`;
- `validation_gain_per_parameter` virou metrica obrigatoria;
- melhor ponto DRM-SAINT-G e selecionado automaticamente por ganho de validacao por
  parametro;
- comparacao contra LoRA rank 1 inclui estimativa por parametro equivalente;
- corrigido bug onde `train_only + measure_loss` podia registrar
  `validation_loss = 0.0` sem validacao real.

Resultado principal, Qwen2.5-14B, `v_proj`, layer 2:

| budget | validation delta | ganho val/param | params |
|---:|---:|---:|---:|
| 16 | +0.011784 | 0.000000e+00 | 16 |
| 32 | -0.006461 | 2.019107e-04 | 32 |
| 64 | +0.013001 | 0.000000e+00 | 64 |

Comparacao:

| metodo | validation delta | ganho val/param | params |
|---|---:|---:|---:|
| DRM-SAINT-G budget 32 | -0.006461 | 2.019107e-04 | 32 |
| LoRA rank 1 forward-hook | -0.004622 | 7.521982e-07 | 6144 |

Mais camadas, budget 32:

| camada | validation delta | ganho val/param |
|---:|---:|---:|
| 1 | -0.019947 | 6.233305e-04 |
| 2 | -0.006461 | 2.019107e-04 |
| 3 | +0.002833 | 0.000000e+00 |

Veredito:

```text
Fase 15 passou como prova de eficiencia extrema por parametro.
Ainda nao passou como substituto geral de LoRA em qualidade absoluta.
```

Direcao tecnica:

```text
Delta W = A Phi B
```

`Phi` deve virar o objeto DRM-SAINT-G principal: operador geometrico, manifold local,
microbloco relacional 2x2/4x4 ou codebook topologico multi-escala.

Proximo marco:

- implementar `DRM-SAINT-G_phi_delta`;
- comparar contra `layer 1 v_proj budget 32`;
- medir ganho por parametro, tamanho de checkpoint e pico CUDA;
- decidir se a proxima escala usa o formato atual ou o formato `Phi`.

### Marco 13 - DRM-SAINT-G Phi Delta

Status: **concluido**.

Mudancas:

- criado `activation_phi_validation_rerank`;
- criado `saint/adapters/huggingface_phi_routing.py`;
- criado `scripts/benchmark_huggingface_phase15_phi.py`;
- `Delta W = A Phi B` foi implementado com SVD local por bloco:
  `A = U_r`, `B = V_r^T`;
- `Phi` e treinavel e o delta denso nao e materializado;
- variantes testadas:
  `dense`, `diagonal`, `upper_triangular`, `block_diagonal`, `kronecker`,
  `hadamard`, `codebook_2x2`, `codebook_4x4`.

Resultado Qwen2.5-14B, `layer 1 v_proj`, budget 32:

| Phi | validation delta | ganho val/param | params |
|---|---:|---:|---:|
| dense | +0.000049 | 0.000000e+00 | 32 |
| diagonal | +0.003608 | 0.000000e+00 | 32 |
| upper_triangular | +0.013865 | 0.000000e+00 | 30 |
| block_diagonal | +0.024908 | 0.000000e+00 | 32 |
| kronecker | +0.000049 | 0.000000e+00 | 32 |
| hadamard | -0.023148 | 7.715861e-04 | 30 |
| codebook_2x2 | +0.004420 | 0.000000e+00 | 32 |
| codebook_4x4 | -0.012781 | 3.993958e-04 | 32 |

Comparacao:

| metodo | validation delta | ganho val/param | params |
|---|---:|---:|---:|
| Marco 12 structured block | -0.019947 | 6.233305e-04 | 32 |
| Marco 13 Phi hadamard | -0.023148 | 7.715861e-04 | 30 |

Veredito:

```text
DRM-SAINT-G Phi superou o melhor ponto do Marco 12.
O melhor Phi foi hadamard, sugerindo que estrutura vence capacidade bruta nesse budget.
```

Proximo marco:

- repetir `Phi=hadamard` com seeds 31, 32 e 33;
- testar layers 1, 2 e 3;
- testar `phi_rank` 2, 4 e 8;
- comparar contra LoRA rank 1;
- inicializar `Phi` por gradiente/ativacao;
- aumentar gradualmente textos de treino e validacao.

### Marco 14 - Confirmacao do Phi Hadamard

Status: **concluido como confirmacao parcial**.

Resultados por seed, `layer 1`, `phi_rank=4`:

| seed | validation delta | ganho val/param |
|---:|---:|---:|
| 31 | -0.009962 | 3.320694e-04 |
| 32 | -0.016432 | 5.477269e-04 |
| 33 | -0.008009 | 2.669652e-04 |

Resultados por layer, `seed=31`, `phi_rank=4`:

| layer | validation delta | ganho val/param |
|---:|---:|---:|
| 1 | -0.016432 | 5.477269e-04 |
| 2 | -0.014893 | 4.964193e-04 |
| 3 | -0.016785 | 5.594889e-04 |

Resultados por rank, `seed=31`, `layer=1`:

| phi_rank | validation delta | ganho val/param |
|---:|---:|---:|
| 2 | +0.022626 | 0.000000e+00 |
| 4 | -0.008009 | 2.669652e-04 |
| 8 | -0.023454 | 7.817904e-04 |

Controle LoRA rank 1:

| metodo | validation delta | ganho val/param | params |
|---|---:|---:|---:|
| DRM-SAINT-G Phi hadamard rank 8 | -0.032917 | 1.097218e-03 | 30 |
| LoRA rank 1 forward-hook | +0.004285 | 0.000000e+00 | 6144 |

Veredito:

```text
Phi hadamard rank 4 e positivo, mas nao bate Marco 12.
Phi hadamard rank 8 e o novo candidato principal.
```

`phi_source=score` nao ajudou nesta rodada; SVD local do peso continuou melhor.

Proximo marco:

- repetir `phi_rank=8` em seeds 31, 32 e 33;
- comparar LoRA rank 1 nas mesmas seeds;
- aumentar textos de treino/validacao;
- testar steps 8 e scheduler;
- testar `gradient_phi_validation_rerank`;
- comparar `v_proj`, `o_proj` e `v_proj + o_proj`;
- decidir se Fase 15 fecha com `Phi hadamard rank 8`.

### Marco 15 - Robustez do Phi Rank 8

Status: **concluido com ressalvas**.

Config principal:

```text
model: Qwen2.5-14B
target: layer 1 v_proj
phi_variant: hadamard
phi_rank: 8
budget: 32
steps: 4
train_texts: 4
validation_texts: 8
```

Resultados multi-seed:

| metodo | mean validation delta | mean ganho val/param | params |
|---|---:|---:|---:|
| DRM-SAINT-G Phi rank 8 | -0.012927 | 4.309018e-04 | 30 |
| LoRA rank 1 | -0.006209 | 1.010563e-06 | 6144 |

Outros testes:

| teste | resultado |
|---|---|
| steps 8 + lr_decay 0.9 | piorou validacao |
| gradient_phi_validation_rerank | OOM no routing: 29.672 GB |
| o_proj | sem ganho |
| v_proj + o_proj | igualou v_proj, nao melhorou |

Veredito da Fase 15:

```text
Fase 15 fecha com Phi hadamard rank 8 como baseline historico para a proxima escala.
```

Ressalva:

```text
DRM-SAINT-G venceu em ganho por parametro e checkpoint pequeno.
Ainda nao venceu LoRA como qualidade absoluta geral.
```

Baseline para a proxima escala:

```text
routing: activation_phi_validation_rerank
phi_variant: hadamard
phi_rank: 8
phi_source: weight
target: v_proj
budget: 32
steps: 4
train_texts: 4
validation_texts: 8
```

## Fase DRM-G - DRM-SAINT-G

Status: **em andamento**.

### Objetivo

Validar crescimento progressivo do `drm_transformer` por enxertos treinaveis,
antes de tentar escalar diretamente para 70B.

Nesta fase, o modelo nao nasce grande. Ele cresce por ciclos:

```text
DRM pequeno
  -> congelar nucleo estavel
  -> anexar modulo novo
  -> treinar enxerto com DRM-SAINT-G-Phi
  -> validar perda global
  -> consolidar, manter ou descartar enxerto
```

### Definicao

DRM-SAINT-G e a linha de pesquisa em que DRM-SAINT-G-Phi deixa de ser apenas um delta
compacto sobre pesos existentes e passa a ser um mecanismo de crescimento:

```text
Delta W = A Phi B
```

onde `Phi` representa um operador local treinavel, geometricamente estruturado,
e o sufixo `G` marca **grafting**, ou enxerto progressivo.

### Entregas

- especificacao de enxerto para DRM;
- API para congelar nucleo e anexar modulos novos;
- inicializacao de enxertos por Phi hadamard rank 8;
- checkpoint separado por enxerto;
- consolidacao de enxertos aprovados;
- criterio para descartar enxertos que pioram validacao;
- benchmark DRM pequeno com crescimento progressivo;
- comparacao contra treino direto do mesmo numero de parametros novos.

### Perguntas

- Um DRM pequeno consegue melhorar por enxertos sem retreinar tudo?
- O enxerto preserva capacidades antigas ou causa esquecimento?
- DRM-SAINT-G-Phi e suficiente para treinar o modulo novo com poucos parametros?
- Consolidar enxertos melhora ou degrada a loss global?
- O custo por ganho e melhor que simplesmente aumentar uma camada e treinar full?

### Criterio de conclusao

A fase passa se DRM-SAINT-G demonstrar um ciclo completo:

```text
modelo DRM pequeno
  -> enxerto novo
  -> treino DRM-SAINT-G-Phi
  -> validacao melhor que base congelada
  -> checkpoint recomponivel
  -> consolidacao sem regressao clara
```

Com isso, a nova Fase 16 pode comparar um full controlado contra grafting antes
da Fase 17 de 70B.

### Marco 1 - Enxerto Phi no DRM 3.5M

Status: **concluido como smoke inicial**.

Implementacao:

- metodo runtime `drm_g_saint_phi_graft`;
- adapter `saint.adapters.drm_grafting`;
- config `configs/drm_g_marco1_3_5m.json`;
- baseline externo `drm_transformer/configs/baselines/small_3.5M.yaml`;
- nucleo DRM congelado;
- enxerto `hidden' = hidden + hidden A Phi B` em `final_norm`;
- 64 parametros treinaveis com `phi_rank=8`.

Resultado:

| metrica | valor |
|---|---:|
| parametros do DRM base | 3.468.581 |
| parametros treinaveis | 64 |
| base_loss | 10.825858 |
| graft_loss | 10.822562 |
| validation_gain | 0.003296 |
| validation_gain_per_parameter | 5.1498e-05 |
| dense_budget_loss | 10.793698 |

Leitura:

O primeiro ciclo DRM-G roda sobre a arquitetura real 3.5M, congela o nucleo e
treina apenas `Phi`. O smoke passa porque melhora a loss da base congelada e
gera checkpoint do runtime. Ainda nao vence a baseline densa de mesmo budget,
entao o Marco 2 deve melhorar `A` e `B` com ativacao/gradiente e usar validacao
menos sintetica.

### Marco 2 - Enxerto com Ativacao/Gradiente

Status: **concluido como smoke inicial**.

Implementacao:

- inicializacao de `A` e `B` por `activation`, `gradient` ou
  `activation_gradient`;
- `target_module` configuravel;
- treino e validacao em batches separados;
- config oficial `configs/drm_g_marco2_3_5m_gradient_block1.json`.

Melhor ponto do smoke:

```text
target_module: blocks.1
projection_init: gradient
phi_rank: 8
learning_rate: 0.005
trainable_params: 64
```

Resultado:

| metrica | valor |
|---|---:|
| base_loss validacao | 10.835747 |
| graft_loss validacao | 10.834869 |
| validation_gain | 0.000877 |
| validation_gain_per_parameter | 1.3709e-05 |
| dense_budget_gain | -0.000225 |

Leitura:

O Marco 2 passou como smoke porque o enxerto `A Phi B` inicializado por gradiente
melhorou validacao em batch separado e venceu a baseline densa de mesmo budget
no ponto `blocks.1`. Ainda precisa de mais seeds, mais exemplos e checkpoint
recomponivel do enxerto antes de fechar a Fase DRM-G.

### Marco 3 - Checkpoint, Consolidacao e Decisao do Enxerto

Status: **infraestrutura implementada; qualidade real ainda rejeitada**.

Implementacao:

- payload real `graft.drm-g.json` com `A`, `Phi`, `B`, `target_module` e `scale`;
- formato `drm_graft_payload_json`;
- metodo runtime `drm_g_saint_phi_eval`;
- config `configs/drm_g_marco3_eval_payload.json`;
- leitura opcional de tokens reais em `drm_transformer/data/baseline`;
- estado real do AdamW do enxerto em `optimizer.saintopt`;
- payload de consolidacao em `consolidation.drm-g.json`;
- `delta_weight` para alvos lineares compativeis;
- decisao automatica `approve/reject` por ganho de validacao;
- sweep `scripts/benchmark_drm_g_marco3.py`.

Resultado de recomposicao:

| metrica | valor |
|---|---:|
| base_loss | 10.835747 |
| graft_loss recomposto | 10.834869 |
| validation_gain | 0.000877 |
| gain/param | 1.3709e-05 |

Melhor ponto do sweep multiseed curto:

| seed | alvo | init | validation_gain | dense_gain |
|---:|---|---|---:|---:|
| 32 | `final_norm` | `activation` | 0.001096 | -0.000362 |
| 32 | `blocks.1` | `gradient` | 0.000933 | -0.000139 |
| 31 | `blocks.2` | `gradient` | 0.000813 | 0.000028 |

Smoke em dados reais tokenizados:

| config | alvo | consolidacao | validation_gain | decisao |
|---|---|---|---:|---|
| `drm_g_marco3_real_tokens.json` | `blocks.1` | hook required | -0.001091 | reject |
| `drm_g_marco3_consolidated_linear.json` | `blocks.1.attn.out_proj` | state delta | -0.000483 | reject |

Leitura:

O checkpoint do enxerto ja e recomponivel e agora tambem salva optimizer real,
payload de consolidacao e criterio automatico de aprovacao. Em dados reais
tokenizados, os dois smokes atuais foram rejeitados. O Marco 3 deixa de ser
apenas "salvar o enxerto" e passa a filtrar enxertos ruins antes de consolidar.

### Marco 4 - Crescimento Progressivo

Status: **passou no criterio minimo em smoke real**.

Objetivo:

```text
DRM base -> DRM + G1 -> DRM + G1 + G2
```

Implementado:

- metodo `drm_g_saint_phi_progressive`;
- config `configs/drm_g_marco4_progressive_real_tokens.json`;
- checkpoint com `drm_graft_sequence_payload`;
- decisao `approve/reject` por enxerto;
- reaplicacao acumulada de enxertos aprovados;
- eval recomposto de sequencia com `drm_g_saint_phi_eval`;
- comparacao por etapa contra `DenseBudgetGraft`.

Resultado:

| etapa | alvo | init | validation_gain | dense_gain | decisao |
|---:|---|---|---:|---:|---|
| 1 | `blocks.2` | `activation` | 0.000338 | -0.000303 | approve |
| 2 | `final_norm` | `activation` | 0.000477 | -0.000107 | approve |

Resumo:

| metrica | valor |
|---|---:|
| base_loss | 10.792871 |
| final_loss | 10.792057 |
| sequence_gain | 0.000814 |
| enxertos aprovados | 2 |
| parametros treinaveis | 128 |

Benchmark multiseed:

| metrica | valor |
|---|---:|
| runs | 8 |
| runs positivos | 4 |
| runs com 2+ enxertos aprovados | 2 |
| melhor sequence_gain | 0.000495 |
| melhor gain/param | 3.8669e-06 |
| melhor checkpoint | 78786 bytes |
| routing_s CPU | 0.2380 |
| train_s CPU | 0.5528 |
| eval_s CPU | 1.2775 |

Melhor run:

| seed | sequencia | aprovados | rejeitados | adiados | old_regression |
|---:|---|---:|---:|---:|---:|
| 33 | `linear_stack` | 2 | 0 | 1 | -0.000495 |

Etapas:

| etapa | alvo | validation_gain | dense_gain | decisao |
|---:|---|---:|---:|---|
| 1 | `blocks.1.attn.out_proj` | 0.000293 | -0.040218 | approve |
| 2 | `blocks.2.attn.out_proj` | 0.000202 | -0.041443 | approve |
| 3 | `blocks.3.attn.out_proj` | -0.000020 | -0.039980 | defer |

Merge permanente:

| metrica | valor |
|---|---:|
| state_dict_merge_supported | true |
| merge_loss_abs_diff | 0.0 |
| merged_graft_loss | 10.804946 |

CUDA:

| metrica | valor |
|---|---:|
| approved_grafts | 3 |
| sequence_gain | 0.000379 |
| cuda_routing_peak_bytes | 71483904 |
| cuda_train_peak_bytes | 51806720 |
| cuda_eval_peak_bytes | 45419520 |

Veredito:

O Marco 4 passa no criterio de benchmark pequeno: ha repeticao multiseed,
tokens reais em mais de um batch, sequencia linear consolidavel, politica
`approve/reject/defer`, medicao de conflito, checkpoint recomponivel e ganho por
parametro contra baseline full-budget linear. O merge permanente em memoria
reproduz a loss via hook exatamente e o smoke CUDA mede picos por etapa.

### Marco 5 - Validacao Robusta e Artefato Consolidado

Status: **iniciado**.

Subfases:

| subfase | foco | saida esperada |
|---|---|---|
| 5A | artefato consolidado em disco | concluido: `.pt` salvo, recarregado e avaliado |
| 5B | retencao e dados maiores | concluido: 4/4 seeds passaram retencao |
| 5C | baseline full mais forte | concluido parcial: competitivo na media, perdeu melhor caso |
| 5D | segundo tamanho DRM | concluido parcial: multilingual 5M |
| 5E | criterio automatico final | concluido: `partial_pass`, 7/8 eixos |
| 5F | relatorio final DRM-G | concluido: avancar para Fase 16 com ressalvas |

Criterio final:

```text
treinar -> aprovar -> consolidar -> salvar .pt -> recarregar -> avaliar -> decidir
```

O Marco 5 fecha quando o artefato consolidado for reproduzivel, a retencao for
medida em mais exemplos, o baseline full for mais forte e houver uma decisao
automatica para avancar ou segurar a fase.

Marco 5A inicial:

| metrica | valor |
|---|---:|
| artifact_bytes | 13902349 |
| saved_loss_abs_diff | 0.0 |
| validation_gain | 0.000495 |
| state_dict_merge_supported | true |

O `.pt` consolidado foi salvo em
`runs/drm_g_marco5a_consolidate_linear/consolidated_model.pt` e reproduziu a
loss do caminho por hook.

Marco 5B inicial:

| metrica | valor |
|---|---:|
| seeds | 31, 32, 33, 34 |
| validation_batches | 8 |
| positive_runs | 4 / 4 |
| retention_passed_runs | 4 / 4 |
| max_old_regression | 0.0002 |
| best_gain_per_parameter | 1.871958e-06 |

O benchmark `scripts/benchmark_drm_g_marco5b.py` confirmou ganho positivo em
todas as seeds testadas e `old_regression` negativa em todos os runs. O resultado
fortalece a retencao no fixture tokenizado, mas ainda nao fecha a escala de dados
exigida para encerrar o Marco 5 inteiro.

Marco 5C inicial:

| metrica | DRM-SAINT-G 4096 | full_budget 4096 | full_module 4096 |
|---|---:|---:|---:|
| validation_gain | 0.000400 | -0.005331 | 0.025951 |
| gain_per_parameter | 9.770156e-08 | -1.301611e-06 | 6.335787e-06 |
| trainable_parameters | 4096 | 4096 | 4096 |
| train_s | 0.156 | 0.114 | 0.081 |

O benchmark `scripts/benchmark_drm_g_marco5c.py` tornou o controle mais forte:
`DRM-SAINT-G` tambem foi testado com 4096 parametros (`phi_rank=64`). Ele venceu
o `full_budget_linear_4096`, mas perdeu por margem grande para
`full_module_linear` no mesmo alvo e com o mesmo numero nominal de parametros.
O Marco 5C ficou implementado, mas nao passou no criterio de qualidade contra
full-module.

Marco 5C - teste de hipoteses A Phi B:

| metodo | mean_gain | mean_gain/param | wins |
|---|---:|---:|---:|
| `phi_zero_4096` | 0.017202 | 4.199748e-06 | 4 / 4 |
| `phi_ls_train_ab` | 0.016895 | 3.299810e-06 | 4 / 4 |
| `full_module_linear` | 0.016382 | 3.999550e-06 | 3 / 4 |
| `phi_ls_residual_4096` | 0.015902 | 3.943040e-06 | 4 / 4 |
| `phi_ls_4096` | 0.015894 | 3.880414e-06 | 4 / 4 |

O benchmark `scripts/benchmark_drm_g_marco5c_phi_variants.py` testou:
least-squares/projecao de gradiente, `Phi + sparse_residual`, `A/B`
parcialmente treinaveis e grid 4096 contra `full_module_linear`. As variantes
Phi ficaram competitivas na media multiseed e tiveram ganho positivo em 4/4
seeds, mas o melhor caso absoluto ainda foi `full_module_linear` na seed 33.
O resultado sugere continuar a parametrizacao `A Phi B`, mas com criterio
separando media multiseed, melhor caso e custo de memoria.

Status formal: 5C fica como **implementado e parcialmente suportado**. Nao houve
vitoria absoluta contra `full_module_linear`, mas houve sinal positivo em media
multiseed, estabilidade 4/4 nas variantes Phi e vantagem contra
`full_budget_linear_4096`.

Proximos pontos levados para 5E/5F:

- separar `best_case_win`, `mean_multiseed_win` e `stability_win`;
- medir `checkpoint_size_win`, `memory_win` e `compression_win`;
- testar `phi_train_ab` com orcamento capado em 4096;
- testar Phi multi-stage `A1 Phi1 B1 + A2 Phi2 B2`;
- medir retencao das variantes Phi como no Marco 5B.

Marco 5D inicial no DRM multilingual 5M:

| metodo | mean_gain | mean_gain/param | positivos | params |
|---|---:|---:|---:|---:|
| `phi_ls_train_ab_half_rank` | 0.009759 | 8.471776e-07 | 3 / 4 | 11520 |
| `phi_zero_full_rank` | 0.009285 | 1.007526e-06 | 3 / 4 | 9216 |
| `phi_ls_full_rank` | 0.009093 | 9.866231e-07 | 3 / 4 | 9216 |
| `phi_ls_residual_full_rank` | 0.009078 | 9.952688e-07 | 3 / 4 | 9121 |
| `full_module_linear` | 0.003337 | 3.621034e-07 | 2 / 4 | 9216 |

O Marco 5D usa `drm_transformer/configs/scaling/multilingual/5m.yaml`. O
benchmark passou operacionalmente e mostrou Phi vencendo `full_module_linear` na
media multiseed e em estabilidade de runs positivos. O melhor caso individual
ainda foi `full_module_linear`, entao o status continua parcialmente suportado,
nao vitoria total.

Marco 5E inicial:

| eixo | passou |
|---|---:|
| `artifact_reproducible` | true |
| `retention_win` | true |
| `best_case_win` | false |
| `mean_multiseed_win` | true |
| `stability_win` | true |
| `checkpoint_size_win` | true |
| `memory_win` | true |
| `compression_win` | true |

O avaliador `scripts/benchmark_drm_g_marco5e.py` gerou
`runs/drm_g_marco5e_phase_decision/phase_decision.json`. O status automatico foi
`partial_pass`: 7 eixos passaram, com falha explicita em `best_case_win`. Este e
o veredito correto para o Marco 5 ate agora: avancar como evidencia parcial, sem
alegar dominancia absoluta contra full-module.

Marco 5F final:

```text
relatorio: docs/reports/drm_g_marco5f_relatorio_final.md
status: concluido
veredito: avancar para Fase 16 como partial_pass
baseline recomendado: drm_transformer/configs/scaling/multilingual/5m.yaml
foco recomendado: infraestrutura e escala controlada
```

O Marco 5F fecha a fase DRM-G com recomendacao tecnica clara: seguir para Fase
16, usando Phi como familia principal, mas mantendo `full_module_linear` como
baseline obrigatorio e sem declarar vitoria absoluta de qualidade.

## Fase 16 - DRM Full 125M/350M vs DRM-SAINT-G Grafted

Status: **pendente**.

### Objetivo

Criar uma ponte cientifica antes da escala 70B: treinar um DRM full que caiba em
uma RTX 4090 e comparar contra um DRM menor crescido por grafting.

Esta fase existe porque a comparacao full em 70B nao e viavel no hardware alvo.
O baseline full precisa existir em uma escala menor usando configs reais do
`drm_transformer`: 125M primeiro, 350M se couber.

### Hipotese

```text
DRM full 125M
DRM full 350M, se couber
vs
DRM 5M + DRM-SAINT-G grafted ate capacidade/budget semelhante
vs
modelos externos pequenos quando aplicavel
```

Nao criar uma config artificial de 250M nesta fase. A comparacao deve usar os
YAMLs existentes `125m.yaml` e `350m.yaml`, com datasets separados derivados do
mesmo superset multilingual.

### Comparacoes

- DRM full controlado;
- DRM 5M congelado;
- DRM 5M + `phi_zero_full_rank`;
- DRM 5M + `phi_ls_residual`;
- DRM 5M + `phi_ls_train_ab` com budget capado;
- GPT-2 124M e/ou GPT-2 Medium 355M quando couber;
- OPT-125M e/ou OPT-350M quando couber.

Observacao:

A comparacao externa deve ser por faixa de tamanho e usada como calibracao de
perplexity, nao como prova direta.

### Marcos

1. Memory planner para DRM full 125M/350M.
2. Config DRM full controlada no maior tamanho que couber na RTX 4090.
3. Treino full curto com loss/perplexity/checkpoint.
4. DRM 5M + grafts ate budget/capacidade alvo.
5. Comparacao full vs grafted.
6. Comparacao externa contra GPT/OPT pequenos.
7. Decisao de entrada para 70B.

Marco 4 inicial:

```text
script: scripts/benchmark_drm_g_phase16_5m_graft.py
relatorio: docs/reports/phase16_marco4_5m_graft.md
melhor alvo: blocks.5.ffn.down_proj
melhor media: phi_ls_full_rank
ganho medio: 0.000114
positivo: 3/3 seeds
params Phi: 9.216
controle full_module_linear: 36.864 params, ganho medio negativo
```

Veredito: o caminho grafted 5M esta operacional e tem sinal positivo inicial,
mas o ganho ainda e pequeno frente ao gap para o full 125M.

Marco 4B - `5M + 5M graft * 24`:

```text
script: scripts/benchmark_drm_g_phase16_graftblock.py
relatorio: docs/reports/phase16_marco4b_graftblock_5m.md
base params: 5.699.059
24 grafts: 119.296.536 params
total efetivo: 124.995.595 params
pico CUDA smoke: 3.43 GB
ganho validacao smoke: 0.000261
```

Veredito: a forma correta do experimento `5M + enxertos ate 125M` agora existe
e roda. O sinal inicial e positivo, mas pequeno. O caso com 4 grafts teve maior
ganho no smoke curto, entao o proximo marco deve treinar 24 grafts de forma
progressiva e comparar 4/8/16/24 com o mesmo budget de tokens.

Marco 4C - treinabilidade 24-graft:

```text
script: scripts/benchmark_drm_g_phase16_graftblock.py
relatorio: docs/reports/phase16_marco4c_graftblock_training.md
full 125M smoke loss: 9.049912
melhor 4-graft loss: 10.415268
24-graft checkpoint positivo loss: 10.415910
24-graft distance to full 125M: 1.365997
checkpoint recomponivel: recompose_abs_diff 0.0
```

Veredito: infraestrutura aprovada, qualidade ainda nao. O 24-graft cabe em CUDA
e gera checkpoint recomponivel, mas mais steps com progressivo ingenuo pioraram
a validacao. Proximo passo: selecao/aceite de enxertos por ganho de validacao,
congelando grupos aceitos antes de adicionar novos.

Nota de eficiencia:

Mesmo sem vencer o full 125M em loss absoluta, o resultado ja e uma vitoria de
eficiencia operacional. O full 125M smoke levou horas para 100 steps, enquanto o
caminho `DRM 5M + 24 grafts ~= 125M` executa o smoke em poucos minutos ou menos,
com pico CUDA de aproximadamente `3.43 GB`, checkpoint recomponivel e perda de
validacao controlada em testes curtos. O gap contra o full 125M ainda existe,
mas a diferenca de custo justifica um teste longo de mesmo tempo de parede:

```text
comparar full 125M 100 steps em ~4h
vs
DRM 5M + 24 grafts treinado por ~4h
```

Esse teste deve medir se a vantagem de velocidade transforma o grafted em uma
alternativa competitiva quando ambos recebem o mesmo tempo real de treino.

### Criterio de sucesso

Sucesso minimo:

```text
treinar DRM full controlado na RTX 4090
treinar DRM 5M + grafts
comparar loss/perplexity/memoria/checkpoint
gerar relatorio final
```

Sucesso forte:

```text
DRM 5M + grafts fica competitivo contra DRM full controlado
em perplexity por byte, memoria ou parametro treinavel
```

### Documento

```text
docs/process/fase_16_full_125m_350m_vs_grafted.md
```

## Fase 17 - Escala 70B

Status: **pendente**.

### Objetivo

Validar DRM-SAINT-G como adaptacao extrema em hardware limitado.

### Condicoes

Nao e treino full de 70B.

E:

```text
modelo base quantizado/congelado
deltas DRM-SAINT-G esparsos
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

### Documento

```text
docs/process/fase_17_escala_70b.md
```

## Trilha Transversal - Escalabilidade em Clusters de GPU

Status: **pendente**.

### Objetivo

Projetar DRM-SAINT-G para escalar alem de uma unica RTX 4090, usando clusters
de GPU de forma parecida com laboratorios grandes e big techs, mas preservando a
ideia central do projeto: treinar capacidade nova por partes, nao todo o modelo
de uma vez.

### Hipotese

DRM-SAINT-G pode aproveitar clusters de GPU em dois niveis:

```text
paralelismo de modelo/dados
e
paralelismo de busca por enxertos
```

O primeiro e o caminho classico:

- DDP;
- FSDP;
- ZeRO/offload quando disponivel;
- gradient checkpointing;
- mixed precision;
- sharding de checkpoints;
- pipeline de dados por shards.

O segundo e especifico do DRM-SAINT-G:

- varias GPUs testam candidatos de graft em paralelo;
- cada job mede ganho de validacao por parametro/byte/tempo;
- uma fila central aprova, rejeita ou adia enxertos;
- enxertos aprovados sao consolidados em checkpoints recomponiveis;
- conflitos entre enxertos sao medidos antes do merge permanente.

### Escala Tecnica

Em uma GPU consumer:

- micro-batch 1;
- roteamento barato;
- deltas esparsos;
- modelo base congelado;
- checkpoint compacto.

Em cluster pequeno:

- uma GPU por candidato de enxerto;
- agregacao de metricas em CPU;
- validacao multiseed paralela;
- sweep de budgets e `Phi` em paralelo.

Em cluster grande:

- FSDP para modelos base grandes;
- shard de optimizer/checkpoints;
- offload CPU/NVMe;
- fila de jobs por camada/matriz;
- merge/consolidacao por etapas;
- relatorio global de eficiencia.

### Metricas

- throughput de tokens/s por GPU;
- tempo de roteamento por candidato;
- tempo de treino por enxerto;
- taxa de candidatos aprovados;
- ganho de validacao por GPU-hora;
- ganho por parametro treinavel;
- ganho por byte de checkpoint;
- memoria maxima por GPU;
- custo de comunicacao;
- regressao apos consolidacao.

### Criterio de Sucesso

A trilha passa quando o projeto demonstrar:

```text
o mesmo experimento DRM-SAINT-G rodando em 1 GPU e N GPUs
com ganho claro de tempo ou cobertura de busca
sem perder recomponibilidade de checkpoint
e com metricas comparaveis por candidato
```

### Risco

Escalar mal tambem e um resultado importante. Se comunicacao, I/O ou roteamento
custarem mais que o treino dos enxertos, DRM-SAINT-G deve priorizar eficiencia
single-GPU antes de perseguir clusters.

## Fase 18 - Otimizacoes

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

O overhead do DRM-SAINT-G deve ficar pequeno o bastante para ser pratico em experimentos reais.

## Fase 19 - Avaliacao

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

- DRM-SAINT-G aprende algo util?
- DRM-SAINT-G destroi capacidades antigas?
- DRM-SAINT-G e mais eficiente que LoRA em algum regime?
- DRM-SAINT-G escala com modelo maior?
- DRM-SAINT-G depende demais do dataset?

## Fase 20 - Produto de Pesquisa

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
drm-saint-g estimate --model tiny
drm-saint-g reconstruct --checkpoint runs/matrix001
drm-saint-g train --config configs/tiny_transformer.yaml
DRM-SAINT-G compare --run runs/DRM-SAINT-G001 --baseline runs/lora001
drm-saint-g merge --run runs/DRM-SAINT-G001 --out merged/
```

## 21. Ordem Recomendada

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
13. full controlado 125M/350M vs grafted
14. 70B
15. escalabilidade em clusters de GPU
```

## 21.1 Proxima Acao Imediata

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
- DRM-SAINT-G perde sempre para LoRA em memoria e qualidade;
- offload torna treino impraticavel;
- consolidacao nao reduz conflito;
- mini-transformer nao converge.

## 20. Marcos

### Marco A - Prova de Matriz

```text
DRM-SAINT-G representa matrizes com codebook multi-escala
e mostra compressao/erro competitivo.
```

### Marco B - Prova de Aprendizado

```text
DRM-SAINT-G treina uma camada linear melhor que uma baseline simples
em pelo menos um regime de memoria.
```

### Marco C - Prova de Acoplamento

```text
DRM-SAINT-G treina um mini-transformer com loss global e atualizacao local.
```

### Marco D - Prova de Modelo Real

```text
DRM-SAINT-G adapta um modelo real pequeno e compara com LoRA.
```

### Marco E - Prova de Escala

```text
DRM-SAINT-G roda em 3B/14B com memoria controlada.
```

### Marco F - Prova Extrema

```text
DRM-SAINT-G executa treino parcial em 70B com checkpoint recomponivel.
```

## 21. Definicao de Sucesso

DRM-SAINT-G sera considerado promissor se demonstrar pelo menos uma destas vantagens:

- menor memoria que LoRA em algum regime;
- menor checkpoint;
- melhor ganho por parametro treinavel;
- melhor ganho por byte de VRAM;
- boa adaptacao com base congelada;
- recomposicao estavel;
- codebook reutilizavel entre camadas ou modelos.

DRM-SAINT-G sera considerado fraco se:

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
