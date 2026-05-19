# DRM-SAINT-G

**DRM-SAINT-G** e um sistema experimental para crescimento e adaptacao extrema
de modelos por enxertos estruturados.

Em vez de assumir que todo avanco precisa treinar todos os pesos de um modelo
gigante, o projeto investiga outra pergunta:

```text
quanto de capacidade nova pode ser adicionado treinando apenas enxertos
compactos, roteados e recomponiveis?
```

O foco inicial e o `drm_transformer`, um backbone proprio baseado em geometria
DRM. A meta de pesquisa e comparar treino full controlado contra crescimento
progressivo por enxertos, antes de tentar escalas como 14B ou 70B.

## Ideia Central

DRM-SAINT-G usa deltas estruturados em vez de fine-tuning completo:

```text
Delta W = A Phi B
```

Onde:

- `A` e `B` projetam o enxerto para a matriz alvo;
- `Phi` e o nucleo treinavel, pequeno e estruturado;
- o modelo base pode ficar congelado;
- os enxertos podem ser aprovados, rejeitados, salvos, recompostos e
  consolidados.

Na pratica, isso transforma o problema:

```text
treinar tudo
```

em:

```text
escolher onde crescer, treinar pouco, medir impacto, consolidar se valer a pena
```

## O Que Este Projeto Nao Promete

DRM-SAINT-G nao promete treinar um 70B do zero em uma RTX 4090.

O objetivo e mais especifico:

- adaptar modelos grandes com deltas pequenos;
- crescer backbones DRM por enxertos progressivos;
- reduzir checkpoint e memoria treinavel;
- medir ganho por parametro treinavel;
- comparar contra LoRA, QLoRA, treino full pequeno e baselines densas.

## Estado Atual

O projeto ja passou por fases de:

- codebooks de blocos;
- roteamento por sensibilidade;
- treino de camada linear;
- mini-transformer;
- checkpoint robusto e shardado;
- adaptador `drm_transformer`;
- modelos Hugging Face pequenos;
- ponte 3B e 14B;
- DRM-G com enxertos `A Phi B`;
- relatorio final DRM-G com recomendacao de avancar com ressalvas.

A fase atual prepara a comparacao:

```text
DRM full 125M/350M
vs
DRM 5M + DRM-SAINT-G grafted
vs
GPT-2/OPT pequenos como calibracao externa
```

## Quick Start

Instale dependencias no ambiente do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Execute a CLI:

```powershell
python -m saint.cli --help
```

Rode testes/smokes pequenos antes de qualquer benchmark maior:

```powershell
python -m pytest
```

## Fluxo Com DRM Transformer

O `drm_transformer` fica em outro reposititorio local. A ponte atual usa as
configs reais:

```text
drm_transformer/configs/scaling/multilingual/125m.yaml
drm_transformer/configs/scaling/multilingual/350m.yaml
```

Prepare primeiro o dataset maior e derive o menor:

```powershell
python scripts/prepare_multilingual_data.py --output-dir data/multilingual_350m --max-tokens 7000000000 --vocab-size 50000 --langs en,pt,es,fr,de

python scripts/prepare_multilingual_data.py --output-dir data/multilingual_350m --vocab-size 50000 --finalize --clean-raw

python scripts/prepare_multilingual_data.py --derive-subset-from data/multilingual_350m --output-dir data/multilingual_125m --max-tokens 3500000000 --subset-copy-mode hardlink
```

Smoke recomendado para o DRM full 125M:

```powershell
python scripts/train_distributed.py `
  --config configs/scaling/multilingual/125m.yaml `
  --device cuda `
  --override batch_size=1 gradient_accumulation_steps=8 total_tokens=819200 save_interval=100 eval_interval=100 log_interval=10 save_dir=checkpoints/multilingual_125m/smoke_100
```

## Conceitos

**Graft**: enxerto treinavel adicionado a uma matriz/camada congelada.

**Phi**: nucleo compacto do enxerto `A Phi B`.

**Roteamento**: escolha das matrizes, blocos ou regioes que receberao capacidade
nova.

**Consolidacao**: incorporacao permanente do enxerto no estado do modelo quando
o ganho justifica.

**Checkpoint recomponivel**: artefato que salva o suficiente para reconstruir o
enxerto sem materializar deltas densos desnecessarios.

## Escalabilidade

DRM-SAINT-G foi desenhado para escalar em duas direcoes:

1. **Uma GPU consumer**: micro-batch, deltas pequenos, checkpoint compacto,
   roteamento barato e modelo base congelado.
2. **Cluster de GPUs**: paralelismo de busca, treino e validacao de enxertos,
   usando ideias semelhantes ao fluxo de big techs: sharding, FSDP/DDP,
   offload, filas de jobs, agregacao de metricas e consolidacao controlada.

Em cluster, o ganho esperado nao vem apenas de aumentar batch size. Vem de
executar muitos candidatos de enxerto em paralelo e promover apenas os que
melhoram validacao por byte, por parametro ou por tempo.

## Roadmap Resumido

1. Fechar a ponte DRM full 125M/350M vs DRM-SAINT-G grafted.
2. Testar 70B como adaptacao parcial, nao treino full.
3. Otimizar runtime, checkpoints e roteamento.
4. Adicionar trilha de escalabilidade em clusters de GPU.
5. Formalizar avaliacao alem de loss: perplexity, retencao, QA simples e
   comparacoes externas.

Detalhes completos ficam em:

```text
docs/roadmap.md
docs/process/
```

## Licenca

Veja `LICENSE`, `COPYRIGHT`, `CLA.md`, `CONTRIBUTING.md`, `SECURITY.md` e
`PRIOR_ART.md`.
