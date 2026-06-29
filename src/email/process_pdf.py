import importlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pikepdf
from dotenv import load_dotenv

from .. import config as cfg
from ..logger import get_logger
from .schemas import Attachment
from .utils import read_manifest, write_manifest

logger = get_logger(__name__, cfg.logging["process_log_file"], cfg.logging["level"])

MANIFEST_PATH = "data/control/email_manifest.json"
PIPELINE_DIR = Path("src/email/pipelines")
OUTPUT_DIR = "data/processed"


def _get_pipeline(bank: str | None, instrument: str | None):
    try:
        return importlib.import_module(f"src.email.pipelines.{bank}_{instrument}")
    except ImportError as e:
        raise ValueError(f"No pipeline registered for {bank}_{instrument}") from e


def _discover_pipelines() -> dict[tuple[str, str], str]:
    registry: dict[tuple[str, str], str] = {}
    if not PIPELINE_DIR.exists():
        return registry
    for f in PIPELINE_DIR.glob("*.py"):
        if f.name.startswith("_"):
            continue
        stem = f.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            bank, instrument = parts
            registry[(bank, instrument)] = f"src.email.pipelines.{stem}"
    return registry


def _process_one(attachment: Attachment, password: str | None = None) -> None:
    module = _get_pipeline(attachment.bank, attachment.instrument)
    attachment.parquet_path = f"{OUTPUT_DIR}/{attachment.attachment_id}.parquet"
    df = module.transform(attachment, password)
    if df is None:
        raise ValueError("dataframe is None")
    df.write_parquet(attachment.parquet_path)


def test_attachment(attachment_id: str) -> None:
    manifest = read_manifest(MANIFEST_PATH)
    test_attachment = [a for a in manifest if a.attachment_id == attachment_id][0]
    _process_one(test_attachment)


def main(mode: str | None = None, max_workers: int = 4) -> list[Attachment]:
    manifest = read_manifest(MANIFEST_PATH)
    candidates = [
        a for a in manifest if a.processing_status not in ("pending", "failed_download")
    ]

    if mode != "full":
        candidates = [a for a in candidates if a.processing_status != "processed"]

    if not candidates:
        logger.info("no attachments to process")
        return manifest

    logger.info(f"processing {len(candidates)} attachments")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for a in candidates:
            password = os.getenv(f"{a.bank}_{a.instrument}")
            futures[executor.submit(_process_one, a, password)] = a

        for future in as_completed(futures):
            attachment = futures[future]
            attachment.parquet_path = f"{OUTPUT_DIR}/{attachment.attachment_id}.parquet"
            try:
                future.result()
                attachment.processing_status = "processed"
                logger.info(f"processed {attachment.attachment_id}")
            except pikepdf.PasswordError as e:
                attachment.processing_status = "failed_extraction"
                attachment.metadata["processing_error"] = str(e)
                attachment.parquet_path = None
                logger.warning(f"password error {attachment.attachment_id}: {e}")
            except Exception as e:
                attachment.processing_status = "failed_processing"
                attachment.metadata["processing_error"] = str(e)
                attachment.parquet_path = None
                logger.error(f"failed {attachment.attachment_id}: {e}")

    write_manifest(manifest, MANIFEST_PATH)
    return manifest


if __name__ == "__main__":
    load_dotenv()
    main(mode="full")
