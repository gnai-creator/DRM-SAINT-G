# Fase 9 - Adaptador DRM Transformer

Status: **concluido**.

## Marco 1 - Integracao e Reconstituicao

Status: **concluido**.

Este marco valida que o runtime SAINT-G consegue operar sobre matrizes reais de um
checkpoint do `drm_transformer`, antes de introduzir treino PyTorch com autograd.

## Implementado

- adapter `drm_transformer` selecionavel por `RuntimeConfig.task`;
- carregamento de checkpoint `.pt` via PyTorch quando disponivel;
- carregamento de checkpoint `.json` para smoke tests dependency-free;
- listagem de matrizes 2D filtradas por palavras-chave;
- mapeamento de matrizes para regioes SAINT-G em blocos;
- metodo `drm_SAINT-G_delta_smoke`;
- geracao de `delta_payload` com deltas pequenos em regioes selecionadas;
- checkpoint SAINT-G com deltas reais;
- merge para `merged_weights`;
- validacao de shapes no merge.

## Fluxo

```text
checkpoint drm_transformer
  -> matrizes 2D
  -> regioes SAINT-G
  -> delta_payload
  -> checkpoint.json
  -> merged_weights
```

## Metodo Runtime

```text
task: drm_transformer
method: drm_SAINT-G_delta_smoke
```

Campos esperados em `metadata`:

```text
checkpoint: caminho do checkpoint
max_dim: limite opcional por matriz
max_matrices: limite opcional de matrizes
block_size: tamanho do bloco SAINT-G
keywords: filtros opcionais de nomes de matriz
```

## Limite do Marco 1

Este marco ainda nao treina o `DRMTransformer` com loss real. Ele valida a parte
estrutural:

```text
inspecionar -> aplicar deltas -> salvar -> retomar -> fundir
```

## Marco 2 - Treino Real com Autograd

Status: **concluido**.

O Marco 2 introduz o caminho PyTorch/autograd para treino pequeno do
`drm_transformer`.

## Implementado no Marco 2

- metodo `drm_SAINT-G_autograd_smoke`;
- criacao de um `DRMTransformer` pequeno;
- loss real de linguagem via cross-entropy do modelo;
- gradientes reais por matriz;
- sensibilidade por bloco com soma de gradiente absoluto;
- selecao de blocos por `parameter_budget`;
- mascara de gradiente para treinar apenas regioes SAINT-G;
- baseline `full` pequeno com o mesmo modelo, dados, steps e learning rate;
- `delta_payload` a partir da diferenca entre pesos finais e pesos iniciais;
- `resume` e `merge` usando os deltas do treino autograd.

## Configuracao Smoke

```text
configs/drm_autograd_smoke.json
```

Comando:

```bash
python -m saint.cli train --config configs/drm_autograd_smoke.json
python -m saint.cli resume --run runs/drm_autograd_smoke
python -m saint.cli merge --run runs/drm_autograd_smoke
```

## Resultado Smoke

```text
initial_loss: 4.1506
SAINT-G_loss: 3.7953
full_baseline_loss: 3.7548
parameter_count: 32
shape_validation: true
```

## Veredito

```text
Fase 9 concluida em escala smoke.
```

O resultado ainda nao prova vantagem do SAINT-G sobre treino full; ele prova que o
runtime agora consegue usar `drm_transformer` com PyTorch/autograd, loss real,
gradientes por bloco, deltas salvos e reconstituicao.
