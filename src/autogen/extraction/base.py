"""Abstract base class for PDF table extractors."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar

from src.autogen.models import ExtractedTable

_DATE_FORMATS = (
    "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y",
    "%d-%b-%Y", "%m/%d/%Y", "%d/%m/%y",
)


def _is_data_cell(cell: str) -> bool:
    """True if cell parses as a date or a number; paren/symbol-decorated labels do not."""
    s = cell.strip()
    if not s:
        return False
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(s, fmt)
            return True
        except ValueError:
            pass
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def promote_header(rows: list[list[str]]) -> list[list[str]]:
    """Drop leading title/junk rows so rows[0] is the column header.

    Header = the label row directly above the first data row (>=1 date/numeric cell).
    Returns rows unchanged when there's no data row, data starts at row 0, or the
    candidate header isn't label-shaped (>=2 populated cells, none data-like).
    """
    first_data = next(
        (i for i, r in enumerate(rows) if any(_is_data_cell(c) for c in r)), None
    )
    if not first_data:  # None or 0 -> nothing above to promote
        return rows
    # Header = nearest label-shaped row above the first data row (>=2 populated
    # cells, none data-like). Scan up to skip stray filler rows between header
    # and the first transaction.
    for i in range(first_data - 1, -1, -1):
        populated = [c for c in rows[i] if c.strip()]
        if len(populated) >= 2 and not any(_is_data_cell(c) for c in rows[i]):
            return rows[i:]
    return rows  # ponytail: no label-shaped header found, keep as-is


def normalize_rows(rows) -> list[list[str]]:
    """Convert every cell to a stripped str. None -> "". Each row -> list[str].

    Accepts list of lists/tuples of arbitrary cell types. Ragged rows are
    preserved as-is.
    """
    result = []
    for row in rows:
        normalized = []
        for cell in row:
            if cell is None:
                normalized.append("")
            else:
                normalized.append(str(cell).strip())
        result.append(normalized)
    return result


class BaseExtractor(ABC):
    name: ClassVar[str]  # each subclass sets a unique short name

    @abstractmethod
    def extract(self, physical_file: str, password: str) -> list[ExtractedTable]:
        """Return tables; empty list is valid.

        Subclasses catch their own library exceptions internally and return []
        (logging the error), EXCEPT password/decryption failures which must
        raise src.autogen.exceptions.ExtractionError.
        """
        ...
