"""AI-assisted ETL pipeline generation."""

from src.autogen.generator import run
from src.autogen.models import GenerationResult

__all__ = ["run", "GenerationResult"]
