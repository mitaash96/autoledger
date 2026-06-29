"""Shared dataclasses for the autogen ETL pipeline generator."""

from dataclasses import dataclass

# Re-export so other autogen modules import Attachment from here.
from src.email.schemas import Attachment

__all__ = [
    "Attachment",
    "ExtractedTable",
    "ExtractionResult",
    "GenerationResult",
    "SampleSet",
    "ScoringResult",
    "TestResult",
    "table_from_dict",
    "table_to_dict",
]


@dataclass
class ExtractedTable:
    name: str | None        # table title if detectable else None
    rows: list[list[str]]   # all cells normalised to stripped strings
    page: int | None        # 1-indexed


@dataclass
class ExtractionResult:
    extractor: str
    attachment_id: str
    tables: list[ExtractedTable]
    error: str | None


@dataclass
class SampleSet:
    dev: list[Attachment]
    test: list[Attachment]


@dataclass
class ScoringResult:
    scores: dict[str, float]                     # extractor_name -> composite score
    per_pdf_scores: dict[str, dict[str, float]]  # attachment_id -> extractor -> score
    winner: str
    winner_score: float
    below_threshold: bool                        # True if winner_score < 0.70


@dataclass
class TestResult:
    __test__ = False  # prevent pytest from collecting this dataclass as a test suite

    attachment_id: str
    name: str
    success: bool
    row_count: int | None
    columns: list[str] | None
    schema_conforms: bool | None
    null_rates: dict[str, float] | None          # column -> null fraction 0.0-1.0
    error: str | None
    parquet_path: str | None
    error_type: str | None = None
    column_dtypes: dict[str, str] | None = None   # column -> polars dtype str (PII-free)


@dataclass
class GenerationResult:
    pipeline_path: str
    winner_extractor: str
    winner_score: float
    codegen_backend: str                         # "local" or "cloud"
    anonymization_applied: bool
    hitl_approved: bool | None                   # None if local path
    test_results: list[TestResult]
    report_path: str
    success: bool


# ---------------------------------------------------------------------------
# Serialization helpers (used by extraction cache)
# ---------------------------------------------------------------------------


def table_to_dict(t: ExtractedTable) -> dict:
    """Serialize an ExtractedTable to a JSON-safe dict."""
    return {
        "name": t.name,
        "rows": t.rows,
        "page": t.page,
    }


def table_from_dict(d: dict) -> ExtractedTable:
    """Deserialize an ExtractedTable from a dict produced by table_to_dict."""
    return ExtractedTable(
        name=d["name"],
        rows=d["rows"],
        page=d["page"],
    )
