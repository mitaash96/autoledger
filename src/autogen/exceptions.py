"""Domain exceptions for the autogen ETL pipeline generator."""


class ExtractionError(Exception):
    """PDF decryption/extraction failure."""


class ScoringError(Exception):
    """All library extractors failed — no codegen target available."""


class CodegenError(Exception):
    """LLM code generation failed (syntax error after retries, API error)."""


class UserAbortError(Exception):
    """HITL review was aborted by the user."""


class PiiLeakError(Exception):
    """Generated code failed the PII safety scan."""
