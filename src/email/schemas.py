import datetime
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, Field

ProcessingStatus = Literal[
    "pending",
    "downloaded",
    "processed",
    "failed_download",
    "failed_extraction",
    "failed_processing",
]


class Attachment(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    attachment_id: str
    raw_attachment_id: str
    name: str
    email_id: str
    processing_status: ProcessingStatus = "pending"
    email_subject: str | None = None
    date: datetime.datetime | None = None
    sender: str | None = None
    physical_file: str | None = None
    bank: str | None = None
    instrument: str | None = None
    parquet_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    dataframe: pl.DataFrame | None = Field(default=None, exclude=True)
