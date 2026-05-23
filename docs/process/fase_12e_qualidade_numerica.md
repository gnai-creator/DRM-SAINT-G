# Fase 12E - Qualidade Numerica

Status: **concluida**.

## Objetivo

Validar se os dtypes compactos preservam a qualidade do modelo apos:

```text
treino -> checkpoint dtype -> merge -> avaliacao de loss
```

## Implementado

- benchmark `benchmark_dtype_quality`;
- treino mini-transformer com `mini_SAINT-G_dynamic_delta`;
- checkpoint por dtype;
- merge por dtype;
- avaliacao de `distillation_loss` nos pesos reconstruidos;
- comparacao contra baseline `float32`;
- teste automatizado do fluxo completo.

## Benchmark

Configuracao:

```text
task: mini_transformer
method: mini_SAINT-G_dynamic_delta
baseline_dtype: float32
dtypes: float32, float16, bfloat16, int8
```

Resultado:

| dtype | bytes | merged_loss | delta loss vs float32 |
|---|---:|---:|---:|
| float32 | 1401 | 0.000123203766 | 0.000000000000 |
| float16 | 944 | 0.000123203746 | -0.000000000020 |
| bfloat16 | 945 | 0.000123204227 | 0.000000000461 |
| int8 | 799 | 0.000123203819 | 0.000000000053 |

## Leitura Tecnica

- `float16` preservou a loss nesse regime e reduziu o checkpoint.
- `bfloat16` tambem preservou a loss, com pequena diferenca numerica.
- `int8` nao degradou a loss neste teste pequeno, mas ainda deve ser tratado
  como experimental ate testes em tarefas maiores.
- A qualidade medida aqui e de um mini-transformer pequeno; a proxima fase deve
  repetir essa validacao em modelos reais pequenos.

## Veredito

```text
Fase 12E concluida.
```

SAINT-G agora tem validacao inicial de qualidade numerica por dtype em uma tarefa
real pequena, fechando a Fase 12.

## Proximo Passo

- Fase 13 - Modelos Hugging Face Pequenos.
