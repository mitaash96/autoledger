from __future__ import annotations

from dotenv.main import load_dotenv

from src.autogen.extraction.runner import extract_one
from src.autogen.models import table_to_dict


def extract_tables(attachment, password):
    """Parse the PDF with the scored extractor.

    Returns a list of tables, each a dict shaped
    ``{"name": str | None, "page": int | None, "rows": list[list[str]]}``
    where rows[0] is the header. This is exactly the shape of the table dicts in
    fixtures/anon_*.json that tables_to_dataframe receives.
    """
    tables = extract_one("docling", attachment.physical_file, password)
    return [table_to_dict(t) for t in tables]


import polars as pl

TARGET_COLUMNS = ["Txn Date", "Narration", "Withdrawals", "Deposits", "Closing Balance"]


def _is_transaction_table(headers):
    lower = [h.lower().strip() for h in headers]
    found = {
        "date": False,
        "narration": False,
        "withdrawal": False,
        "deposit": False,
        "closing_balance": False,
    }
    for h in lower:
        if "txn" in h and "date" in h:
            found["date"] = True
        elif "narration" in h:
            found["narration"] = True
        elif "withdrawal" in h:
            found["withdrawal"] = True
        elif "deposit" in h and "closing" not in h:
            found["deposit"] = True
        elif "closing" in h and "balance" in h:
            found["closing_balance"] = True
    return all(found.values())


def _split_newlines(cell):
    if "\n" in cell:
        return cell.split("\n")
    return [cell]


def _align_rows(data_rows, ncols):
    if not data_rows:
        return []

    best_col = 0
    best_n = 1
    for col_idx in range(ncols):
        max_parts = 1
        for row in data_rows:
            if col_idx < len(row):
                parts = _split_newlines(row[col_idx])
                if len(parts) > max_parts:
                    max_parts = len(parts)
        if max_parts > best_n:
            best_n = max_parts
            best_col = col_idx

    if best_n <= 1:
        return data_rows

    result = []
    for row in data_rows:
        anchor_parts = _split_newlines(row[best_col] if best_col < len(row) else "")
        n = max(best_n, len(anchor_parts))
        aligned = []
        for col_idx in range(ncols):
            cell = row[col_idx] if col_idx < len(row) else ""
            parts = _split_newlines(cell)
            if len(parts) == n:
                aligned.append(parts)
            elif len(parts) == 1:
                aligned.append(parts * n)
            else:
                padded = parts + [""] * (n - len(parts))
                aligned.append(padded)
        for i in range(n):
            result.append([col[i] for col in aligned])
    return result


def tables_to_dataframe(tables):
    all_rows = []
    for table in tables:
        rows = table.get("rows", [])
        if not rows or len(rows) < 2:
            continue
        headers = rows[0]
        if not _is_transaction_table(headers):
            continue
        ncols = len(headers)
        data_rows = [row for row in rows[1:] if any(cell.strip() for cell in row)]
        aligned = _align_rows(data_rows, ncols)
        all_rows.extend(aligned)

    records = []
    for row in all_rows:
        rec = {}
        for i, col in enumerate(TARGET_COLUMNS):
            val = row[i] if i < len(row) else ""
            rec[col] = val.strip()
        records.append(rec)

    if not records:
        return pl.DataFrame({col: [""] for col in TARGET_COLUMNS})

    return pl.DataFrame(records)


def transform(attachment, password):
    return tables_to_dataframe(extract_tables(attachment, password))


