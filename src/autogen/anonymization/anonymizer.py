"""Format-preserving anonymizer for ExtractedTable rows before cloud codegen."""

from __future__ import annotations

import random
import re
import string
from datetime import datetime, timedelta

from src import config as cfg
from src.autogen.models import ExtractedTable
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

_DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y",
    "%d-%b-%Y", "%m/%d/%Y", "%d/%m/%y",
]

_KEYWORDS = {
    "balance", "date", "description", "narration", "credit", "debit",
    "total", "opening", "closing", "account", "statement", "amount",
    "withdrawal", "deposit", "transaction", "branch", "cheque", "ref",
    "value", "particulars",
}

_INDICATORS = {  # txn-type / channel codes that carry parse signal, not PII
    "upi", "neft", "imps", "rtgs", "ach", "nach", "tpt", "pos", "atm",
    "emi", "mb", "ib", "inb", "inf", "chg", "gst", "tds", "cms", "ecs", "int",
}

_NAME_POOL = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Liam", "Maria", "Noah", "Olivia", "Peter",
    "Quinn", "Rose", "Sam", "Tara", "Uma", "Victor", "Wendy", "Yara",
    "Brown", "Chen", "Davis", "Evans", "Fisher", "Garcia", "Harris",
    "Iyer", "Jones", "Kumar", "Lewis", "Moore", "Nair", "Patel",
    "Rao", "Smith", "Taylor", "Umar", "Varma", "White", "Young", "Zhang",
]


class TableAnonymizer:
    """Anonymizes table cells in a format-preserving, deterministic way."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._scale = self._rng.uniform(0.8, 1.2)
        sign = self._rng.choice([-1, 1])
        self._day_offset = timedelta(days=sign * self._rng.randint(30, 180))
        self._code_cache: dict[str, str] = {}
        self._name_cache: dict[str, str] = {}
        self._text_cache: dict[str, str] = {}

    def _detect_type(self, value: str) -> str:
        """Return one of 'date', 'code', 'numeric', 'name', 'text'."""
        s = value.strip()

        for fmt in _DATE_FORMATS:
            try:
                datetime.strptime(s, fmt)
                return "date"
            except ValueError:
                pass

        if s.isdigit() and len(s) >= 9:
            return "code"
        if re.fullmatch(r"[A-Za-z0-9\-/ ]+", s) and any(c.isdigit() for c in s) and any(c.isalpha() for c in s):
            return "code"

        try:
            float(s.replace(",", ""))
            return "numeric"
        except ValueError:
            pass

        tokens = s.split()
        if tokens and all(t.isalpha() and t == t.title() for t in tokens):
            if not any(t.lower() in _KEYWORDS for t in tokens):
                return "name"

        return "text"

    def _anonymize_numeric(self, value: str) -> str:
        s = value.strip()
        has_comma = "," in s
        num = float(s.replace(",", ""))
        decimals = len(s.split(".")[1]) if "." in s else 0
        scaled = abs(num * self._scale)
        fmt = f"{{:,.{decimals}f}}" if has_comma else f"{{:.{decimals}f}}"
        result = fmt.format(scaled)
        return f"-{result}" if num < 0 else result

    def _anonymize_date(self, value: str) -> str:
        s = value.strip()
        for fmt in _DATE_FORMATS:
            try:
                dt = datetime.strptime(s, fmt)
                return (dt + self._day_offset).strftime(fmt)
            except ValueError:
                pass
        return value

    def _anonymize_code(self, value: str) -> str:
        if value in self._code_cache:
            return self._code_cache[value]
        chars = []
        for ch in value:
            if ch.isdigit():
                chars.append(self._rng.choice(string.digits))
            elif ch.isupper():
                chars.append(self._rng.choice(string.ascii_uppercase))
            elif ch.islower():
                chars.append(self._rng.choice(string.ascii_lowercase))
            else:
                chars.append(ch)
        result = "".join(chars)
        self._code_cache[value] = result
        return result

    def _anonymize_name(self, value: str) -> str:
        if value in self._name_cache:
            return self._name_cache[value]
        token_count = len(value.split())
        result = " ".join(self._rng.choice(_NAME_POOL) for _ in range(token_count))
        self._name_cache[value] = result
        return result

    def _scrub_token(self, tok: str) -> str:
        """Mask one separator-free token; keep indicators, keywords, PSP domains, lowercase words."""
        if "@" in tok:  # handle: mask local part, keep domain
            local, _, domain = tok.partition("@")
            return f"{self._scrub_token(local)}@{domain}"
        typ = self._detect_type(tok)
        if typ == "numeric":
            return self._anonymize_numeric(tok)
        if typ == "date":
            return self._anonymize_date(tok)
        if typ == "code":  # has digits → acct/ref/phone
            return self._anonymize_code(tok)
        low = tok.lower()
        if low in _INDICATORS or low in _KEYWORDS:
            return tok  # parse signal, non-PII
        if len(tok) >= 2 and (tok.isupper() or tok.istitle()):
            return self._anonymize_name(tok)  # capitalized → person-name
        return tok  # lowercase common word → keep

    def _anonymize_text(self, value: str) -> str:
        """Scrub free text token-wise: keep parse signals, mask only PII shapes."""
        if value in self._text_cache:
            return self._text_cache[value]
        # ponytail: don't split on ',' — keeps embedded numerics like 1,234.56 intact
        parts = re.split(r"([\s/\-:]+)", value)
        out = [
            p if (not p or re.fullmatch(r"[\s/\-:]+", p)) else self._scrub_token(p)
            for p in parts
        ]
        result = "".join(out)
        self._text_cache[value] = result
        return result

    def anonymize(self, tables: list[ExtractedTable]) -> list[ExtractedTable]:
        """Return new tables with all non-header cells anonymized by detected type."""
        result = []
        masked_cells = 0
        for table in tables:
            new_rows: list[list[str]] = []
            for i, row in enumerate(table.rows):
                if i == 0:
                    new_rows.append(list(row))
                    continue
                new_row = []
                for cell in row:
                    typ = self._detect_type(cell)
                    if typ == "numeric":
                        new_row.append(self._anonymize_numeric(cell))
                        masked_cells += 1
                    elif typ == "date":
                        new_row.append(self._anonymize_date(cell))
                        masked_cells += 1
                    elif typ == "code":
                        new_row.append(self._anonymize_code(cell))
                        masked_cells += 1
                    elif typ == "name":
                        new_row.append(self._anonymize_name(cell))
                        masked_cells += 1
                    else:  # text
                        new_row.append(self._anonymize_text(cell))
                        masked_cells += 1
                new_rows.append(new_row)
            result.append(ExtractedTable(name=table.name, rows=new_rows, page=table.page))
        logger.info("anonymized %d cell(s) masked", masked_cells)
        return result
