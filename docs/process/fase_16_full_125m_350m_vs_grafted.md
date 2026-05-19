# Fase 16 - DRM Full 125M/350M vs DRM-SAINT-G Grafted

Status: **pendente**.

## Objetivo

Criar uma ponte cientifica antes da escala 70B: treinar um modelo DRM full que
caiba em uma RTX 4090 e comparar contra um modelo DRM menor que cresce por
grafting ate budget equivalente.

Esta fase existe porque a comparacao full em 70B nao e viavel no hardware alvo.
Em vez de exigir um baseline impossivel, a Fase 16 cria um baseline full real em
escala controlada.

## Hipotese

Se DRM-SAINT-G e um paradigma util de crescimento por enxerto, entao ele deve
mostrar sinal em uma escala onde ainda conseguimos comparar contra treino full:

```text
DRM full 125M ou 350M
vs
DRM 5M + grafts progressivos ate capacidade/checkpoint equivalente
vs
modelos externos pequenos quando aplicavel
```

O objetivo nao e provar que o grafted vence sempre. O objetivo e medir:

- qualidade por parametro treinavel;
- qualidade por byte de checkpoint;
- estabilidade;
- custo de memoria;
- custo de tempo;
- distancia ate um full model treinado no mesmo dominio.

## Tamanho Alvo

O alvo agora deve seguir os YAMLs reais ja existentes no `drm_transformer`:

```text
drm_transformer/configs/scaling/multilingual/125m.yaml
drm_transformer/configs/scaling/multilingual/350m.yaml
```

O 125M deve ser o primeiro baseline full obrigatorio. O 350M deve ser o segundo
baseline se couber na RTX 4090 com tempo aceitavel.

Para uma RTX 4090, 125M e plausivel para treino experimental. O 350M pode
exigir mais cuidado:

- `bf16` ou `fp16`;
- batch pequeno;
- gradient checkpointing se necessario;
- contexto curto;
- AdamW com cuidado;
- acumulacao de gradiente;
- checkpoints compactos.

Nao vamos criar uma config intermediaria artificial de 250M nesta fase. O Marco
1 deve estimar:

```text
125M
350M
```

O criterio e simples:

```text
o maior DRM full que treina de forma estavel na RTX 4090
sem OOM e com tempo aceitavel
```

## Comparacoes

### Comparacao Principal

```text
DRM full 125M
DRM full 350M, se couber
vs
DRM 5M + DRM-SAINT-G grafted ate 125M/350M nominal/funcional
```

Aqui "ate 125M/350M" nao significa necessariamente materializar todos esses
parametros como treinaveis. Pode significar:

- mesmo budget de inferencia;
- mesmo budget de checkpoint;
- mesmo numero efetivo de parametros adicionados;
- mesma familia arquitetural com capacidade nova por enxertos.

O documento de cada experimento deve declarar qual nocao de equivalencia esta
sendo usada.

### Comparacoes Externas

Comparar perplexity contra modelos externos pequenos e util, mas deve ser feito
com cuidado.

Modelos candidatos:

- `facebook/opt-125m`;
- `facebook/opt-350m`;
- `gpt2` pequeno, 124M;
- `gpt2-medium`, 355M;
- outro causal LM local abaixo de 500M.

Observacao:

A comparacao externa deve ser por faixa real:

```text
~125M
~350M
```

Esses modelos nao sao equivalentes ao DRM, mas ajudam a calibrar perplexity,
tempo e memoria.

## Baselines

Baselines obrigatorios:

- DRM full controlado, maior que caiba;
- DRM 5M congelado;
- DRM 5M + Phi graft;
- DRM 5M + Phi multi-stage se disponivel;
- DRM 5M + full-module graft pequeno;
- modelo externo pequeno sem fine-tuning;
- LoRA/QLoRA externo quando couber.

Baselines opcionais:

- DRM full 125M;
- DRM full 350M se couber;
- grafted a partir de 25M/100M, se houver configs intermediarias.

## Marcos

### Marco 1 - Memory Planner 125M/350M

Objetivo:

Determinar o maior DRM full treinavel na RTX 4090.

Entregas:

- estimativa de parametros por config;
- memoria de pesos;
- memoria de gradientes;
- memoria de AdamW;
- memoria de ativacoes;
- estimativa por dtype;
- recomendacao de execucao: 125M obrigatorio, 350M se couber.

Criterio:

Passa se o projeto escolher uma config full viavel antes de treinar.

### Marco 2 - Config DRM Full Controlada

Objetivo:

Criar config DRM full para o tamanho escolhido.

Entregas:

- config em `drm_transformer/configs/scaling/...`;
- estimativa de parametros;
- script de smoke;
- forward/loss em dataset pequeno;
- medicao CUDA de load e forward.

Criterio:

Passa se o modelo full carrega e roda forward sem OOM.

### Marco 3 - Treino Full Curto

Objetivo:

Treinar o DRM full por poucos steps para criar baseline real.

Entregas:

- treino curto;
- loss antes/depois;
- perplexity em validacao;
- memoria CUDA por etapa;
- tokens/s;
- checkpoint full;
- custo de disco.

Criterio:

Passa se o full model melhora loss sem OOM.

### Marco 4 - Grafted 5M ate Budget Alvo

Objetivo:

Partir de `multilingual/5m.yaml` e adicionar capacidade por DRM-SAINT-G.

Entregas:

- escolher alvos de enxerto;
- aplicar `phi_zero_full_rank`;
- aplicar `phi_ls_residual`;
- testar Phi multi-stage;
- salvar checkpoint recomponivel;
- medir loss/perplexity;
- medir tamanho do checkpoint.

Criterio:

Passa se o grafted melhora a base 5M e gera checkpoint recomponivel.

### Marco 5 - Comparacao Full vs Grafted

Objetivo:

Comparar full controlado contra grafted em metricas honestas.

Metricas:

- validation loss;
- perplexity;
- ganho por parametro treinavel;
- ganho por byte de checkpoint;
- memoria CUDA maxima;
- tempo por step;
- tokens/s;
- estabilidade por seed;
- regressao em exemplos antigos.

Criterio:

Passa se o relatorio deixa claro onde grafting ganha e onde perde.

### Marco 6 - Comparacao Externa

Objetivo:

Comparar perplexity e custo contra modelos externos pequenos.

Entregas:

- avaliar `gpt2` 124M;
- avaliar `gpt2-medium` 355M se couber;
- avaliar `opt-125m`;
- avaliar `opt-350m` se couber;
- mesma tokenizacao/corpus quando possivel;
- relatorio de incompatibilidades de tokenizer.

Criterio:

Passa se a comparacao externa for interpretavel, mesmo que nao seja perfeitamente
equivalente ao DRM.

### Marco 7 - Decisao de Entrada para 70B

Objetivo:

Decidir se vale mover a escala 70B para a fase seguinte.

Perguntas:

- Grafted chega perto do full controlado em perplexity?
- Grafted vence em checkpoint/memoria?
- O custo de tempo e aceitavel?
- O metodo e estavel em mais de uma seed?
- A vantagem permanece em dataset maior?

Criterio:

Passa se houver recomendacao clara:

```text
avancar para Fase 17 - Escala 70B
ou
voltar para melhorar Phi/runtime
```

## Criterio de Sucesso da Fase

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

Falha:

- nenhum DRM full controlado cabe;
- grafted nao melhora a base 5M;
- checkpoint grafted nao recompõe;
- perplexity fica muito pior sem vantagem de memoria/checkpoint;
- comparacao externa fica inutil por incompatibilidade de dados/tokenizer.

## Veredito Sobre a Ideia

A ideia e boa e cientificamente melhor que saltar direto para 70B.

Ela cria uma pergunta testavel:

```text
Se eu tenho hardware para treinar full 125M e talvez 350M,
DRM-SAINT-G consegue crescer de 5M para capacidade parecida
com melhor eficiencia?
```

Isso cria um baseline real. Depois, quando formos para 70B, teremos uma curva de
extrapolacao, nao apenas uma demonstracao isolada.

## Proximo Passo Imediato

Implementar o Marco 1:

- memory planner para DRM full 125M/350M;
- estimativa de parametros das configs candidatas;
- recomendacao automatica do maior tamanho viavel na RTX 4090;
- atualizar roadmap com o tamanho escolhido.
