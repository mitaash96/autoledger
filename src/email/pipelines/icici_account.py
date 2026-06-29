from __future__ import annotations

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


def tables_to_dataframe(tables: list[dict]) -> pl.DataFrame:
    all_rows: list[list[str]] = []

    for table in tables:
        rows = table.get("rows", [])
        if not rows or len(rows) < 2:
            continue

        header = [str(h).upper().strip() for h in rows[0]]

        def _find(*keywords: str) -> int | None:
            for i, h in enumerate(header):
                if any(kw in h for kw in keywords):
                    return i
            return None

        date_idx = _find("DATE", "TXN")
        mode_idx = _find("MODE")
        part_idx = _find("PARTICULAR", "NARRATION", "DESCRIPTION")
        dep_idx = _find("DEPOSIT", "DEPOSITS", "CR", "CREDIT")
        wd_idx = _find("WITHDRAWAL", "WITHDRAWALS", "DR", "DEBIT")
        bal_idx = _find("BALANCE", "CB")

        if date_idx is None or bal_idx is None:
            continue
        if dep_idx is None and wd_idx is None:
            continue

        def _cell(row: list, idx: int | None) -> str:
            if idx is None or idx >= len(row):
                return ""
            return str(row[idx]).strip()

        def _split(c: str) -> list[str]:
            return [s.strip() for s in c.split("\n") if s.strip()] if c else []

        def _norm(c: str, n: int) -> list[str]:
            segs = _split(c)
            if not segs:
                return [""] * n
            if len(segs) == n:
                return segs
            if len(segs) == 1:
                return segs * n
            if len(segs) > n:
                return segs[: n - 1] + [" ".join(segs[n - 1 :])]
            return segs + [""] * (n - len(segs))

        for row in rows[1:]:
            dv = _cell(row, date_idx)
            if not dv:
                if all_rows and part_idx is not None:
                    p = _cell(row, part_idx)
                    if p:
                        prev = all_rows[-1]
                        all_rows[-1] = [
                            prev[0],
                            (prev[1] + " " + p).strip() if prev[1] else p,
                            prev[2],
                            prev[3],
                            prev[4],
                        ]
                continue

            ds = _split(dv)
            if not ds:
                continue
            n = len(ds)

            ms = _norm(_cell(row, mode_idx), n)
            ps = _norm(_cell(row, part_idx), n)
            dep = _norm(_cell(row, dep_idx), n)
            wd = _norm(_cell(row, wd_idx), n)
            bal = _norm(_cell(row, bal_idx), n)

            for i in range(n):
                if not ds[i]:
                    continue
                m, p = ms[i], ps[i]
                narration = ""
                if m and m != ds[i]:
                    narration = m
                if p:
                    narration = (narration + " " + p).strip() if narration else p
                all_rows.append([ds[i], narration, wd[i], dep[i], bal[i]])

    if not all_rows:
        raise ValueError("No transaction data found")

    return pl.DataFrame(
        all_rows,
        schema=["Txn Date", "Narration", "Withdrawals", "Deposits", "Closing Balance"],
        orient="row",
    )


def transform(attachment, password):
    return tables_to_dataframe(extract_tables(attachment, password))


