"""Prompt construction for the codegen LLM."""

import json

from src.autogen.models import ExtractedTable


def trim_tables(tables: list[ExtractedTable], max_tables: int = 15, max_rows: int = 10) -> list[ExtractedTable]:
    """Cap to max_tables (prefer most columns); keep the header plus max_rows data rows."""
    sorted_tables = sorted(tables, key=lambda t: len(t.rows[0]) if t.rows else 0, reverse=True)
    trimmed = []
    for t in sorted_tables[:max_tables]:
        rows = t.rows[: max_rows + 1] if len(t.rows) > 1 else t.rows
        trimmed.append(ExtractedTable(name=t.name, rows=rows, page=t.page))
    return trimmed


def build_prompt(
    bank: str,
    instrument: str,
    winner_extractor: str,
    winner_library: str,
    tables: list[ExtractedTable],
    target_schema: dict[str, str],
    feedback: str | None = None,
) -> str:
    """Build the single user-message prompt string for the codegen LLM."""
    trimmed = trim_tables(tables)
    table_dicts = [{"name": t.name, "page": t.page, "rows": t.rows} for t in trimmed]

    prompt = f"""You are generating a Python ETL script for a personal finance pipeline.

Bank: {bank}
Instrument: {instrument}
Best extractor: {winner_extractor}
Parser library: {winner_library}

Target schema (column -> polars dtype): {json.dumps(target_schema)}

Sample tables extracted from representative PDFs:
{json.dumps(table_dicts, indent=2)}

Requirements — follow ALL of these exactly:
1. Return ONLY Python source — no explanation, no markdown fences, no prose.
2. Implement `transform(attachment, password) -> pl.DataFrame` that reads `attachment.physical_file`, decrypts using `password`, parses the relevant tables with `{winner_library}`, and returns a polars DataFrame.
3. Output DataFrame must have EXACTLY these columns and dtypes: {json.dumps(target_schema)}.
4. Discard irrelevant tables (address blocks, customer metadata, promo content, anything not contributing to the schema).
5. Cells may contain embedded newlines — either wrapped text for one record or several records merged into one cell. Do NOT split each cell on newlines independently (that inflates and misaligns rows). Prefer the parser library's word/cell coordinates (group by y-position into row bands) when available. Otherwise anchor the row count to the column with exactly one value per record (date, else amount): split it on newlines to get N sub-rows, split other columns on newlines and zip positionally when they have N segments, and broadcast a single-segment cell across all N sub-rows. Never emit more rows than the anchor column implies.
6. Imports: only stdlib, `polars` (as `pl`), and `{winner_library}`.
7. No `if __name__ == "__main__"` block. No hardcoded file paths. `password` is a parameter only — never store it.
8. No hardcoded data values, sample rows, account numbers, names, or example records — not in code, comments, or docstrings.
"""
    if feedback and feedback.strip():
        prompt += f"""
# Previous attempt failed — fix these issues and return the corrected full script:
{feedback}"""
    return prompt
