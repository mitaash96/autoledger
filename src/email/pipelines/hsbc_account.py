import polars as pl
from src.email.schemas import Attachment


def transform(attachment: Attachment, password: str | None) -> pl.DataFrame:
    raise NotImplementedError("HSBC account pipeline not yet implemented")
