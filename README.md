# autoledger

*Reconstruct your financial history from bank statements and SMS — with AI-generated, privacy-first ETL pipelines.*

## What it does

Builds a unified personal ledger from password-protected bank-statement PDFs (pulled from email) and SMS alerts. Transactions are cleaned and normalized into fact + dimension tables ready for analysis and visualization.

## Why it's interesting: `src/autogen`

The core of the project is an AI-assisted ETL pipeline generator:

- **Extraction** — benchmarks four PDF table extractors per bank (docling / camelot / pdfplumber / text-layer) and picks the highest-scoring one
- **Anonymization** — a PII guard strips real account numbers, names, and amounts before any data leaves the local machine; a format-preserving anonymizer replaces them with realistic-looking fakes
- **Codegen** — a coding agent (opencode / Ollama / OpenRouter) writes the bank-specific `transform()` function from the anonymized sample, driven by a caller-supplied target schema (`{column: polars_dtype}`); the generated code is contractually required to produce a DataFrame with exactly those columns and types
- **HITL review** — a human-in-the-loop step lets you approve or reject the generated code before it runs on real data
- **Test + report** — generated code is executed in a sandbox; results are surfaced in an HTML report

The pipeline is regenerated per bank, so adding a new bank is: get one statement, run autogen.

## Architecture

```
email/          ingest → download PDFs, parse SMS alerts
autogen/        extract → score → anonymize → codegen → test → report
clean/          normalize transactions into fact + dimension tables
accumulators/   per-bank running totals
dataviz/        D3.js dashboards
```

Orchestrated with **Airflow**, containerized with **Docker**.

## Tech stack

- Python 3.14, `uv` (no conda, no pip directly)
- `polars` — all dataframes; pandas is not used
- Airflow, Docker
- D3.js for visualization
- `basedpyright` as the type checker

## Privacy

**No real financial data is committed to this repository.**

- Bank statements (PDFs) live outside version control
- The `sandbox/` directories and any extraction fixtures are git-ignored
- Credentials and PDF passwords are read from environment variables
- The anonymizer runs before any data is sent to an LLM

## Status

Personal portfolio project — active development.
