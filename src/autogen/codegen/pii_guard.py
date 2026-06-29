"""PII safety scanner for generated ETL source code."""

import re

from src import config as cfg
from src.autogen.exceptions import PiiLeakError
from src.autogen.models import ExtractedTable
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_DIGIT_RUN_RE = re.compile(r"\d{8,}")


def find_violations(source: str, password: str = "", input_tables: list[ExtractedTable] | None = None) -> list[str]:
    """Return a list of human-readable PII findings in generated source (empty = clean)."""
    findings: list[str] = []

    if password and password in source:
        findings.append("Password literal found in generated source.")

    if _EMAIL_RE.search(source):
        findings.append("Email address found in generated source.")

    if _DIGIT_RUN_RE.search(source):
        findings.append("Run of 8+ consecutive digits found in generated source (possible account/reference number).")

    if input_tables:
        leaked: set[str] = set()
        for table in input_tables:
            for row in table.rows[1:]:  # skip header row
                for cell in row:
                    stripped = cell.strip()
                    if len(stripped) >= 6 and stripped in source:
                        leaked.add(stripped)
        for value in sorted(leaked):
            findings.append(f"Verbatim data cell value found in generated source: {value!r}")

    return findings


def scan(source: str, password: str = "", input_tables: list[ExtractedTable] | None = None) -> None:
    """Raise PiiLeakError if find_violations is non-empty; else return None."""
    violations = find_violations(source, password=password, input_tables=input_tables)
    if violations:
        logger.warning("pii scan found %d violation(s)", len(violations))
        raise PiiLeakError("; ".join(violations))
    logger.info("pii scan clean")
