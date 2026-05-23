# Fase 12D - Compatibilidade e Migracao

Status: **concluida**.

## Objetivo

Adicionar uma migracao real de manifesto de checkpoint e validar erro claro para
versoes futuras incompativeis.

## Implementado

- `checkpoint.json` agora usa `format_version: 2`;
- separacao entre versao do manifesto e versao do payload binario;
- campo `compatibility`;
- migracao automatica de manifesto v1 para v2;
- campo `migrated_from` apos migracao;
- erro explicito para versao futura;
- testes de leitura de manifesto antigo;
- testes de rejeicao de manifesto futuro.

## Manifesto v2

Exemplo:

```text
format: SAINT-G_checkpoint
format_version: 2
compatibility:
  min_reader_version: 1
  payload_format_version: 1
  writer_version: 2
files:
  - path: deltas.SAINT-Gbin
    payload: delta
    sha256: ...
```

## Migracao v1 -> v2

A migracao preserva o manifesto antigo e adiciona:

```text
format_version: 2
compatibility:
  min_reader_version: 1
  payload_format_version: 1
  writer_version: 1
migrated_from:
  format_version: 1
  migration: manifest_v1_to_v2
```

## Regra de Compatibilidade

- `format_version == 2`: leitura direta;
- `format_version == 1`: migracao automatica;
- `format_version > 2`: erro antes de validar payloads ou merge;
- versao ausente ou invalida: erro.

## Veredito

```text
Fase 12D concluida.
```

SAINT-G agora possui uma politica explicita de compatibilidade para manifestos de
checkpoint e um caminho testado de migracao `v1 -> v2`.

## Proxima Subfase

- Fase 12E - Qualidade Numerica.
