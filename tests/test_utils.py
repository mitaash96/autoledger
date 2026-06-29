import json
import os
import tempfile

from src.email.schemas import Attachment
from src.email.utils import (
    read_manifest,
    write_manifest,
    refresh_manifest,
    migrate_manifest,
)


def test_read_manifest_returns_empty_when_no_file():
    result = read_manifest("/tmp/nonexistent_manifest.json")
    assert result == []


def test_write_then_read_manifest_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        attachment = Attachment(
            attachment_id="test123",
            raw_attachment_id="raw123",
            name="test.pdf",
            email_id="email123",
            processing_status="downloaded",
            bank="hsbc",
            instrument="account",
        )
        write_manifest([attachment], path)
        result = read_manifest(path)
        assert len(result) == 1
        assert result[0].attachment_id == "test123"
        assert result[0].bank == "hsbc"
    finally:
        os.unlink(path)


def test_migrate_adds_missing_fields():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(
            [
                {
                    "attachment_id": "test123",
                    "raw_attachment_id": "raw123",
                    "name": "test.pdf",
                    "email_id": "email123",
                    "processing_status": "downloaded",
                }
            ],
            f,
        )
        path = f.name

    try:
        result = migrate_manifest(path)
        assert len(result) == 1
        assert result[0].parquet_path is None
    finally:
        os.unlink(path)


def test_migrate_removes_deprecated_fields():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(
            [
                {
                    "attachment_id": "test123",
                    "raw_attachment_id": "raw123",
                    "name": "test.pdf",
                    "email_id": "email123",
                    "processing_status": "downloaded",
                    "old_deprecated_field": "should be removed",
                }
            ],
            f,
        )
        path = f.name

    try:
        migrate_manifest(path)
        raw = json.load(open(path))
        assert "old_deprecated_field" not in raw[0]
    finally:
        os.unlink(path)


def test_migrate_idempotent():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(
            [
                {
                    "attachment_id": "test123",
                    "raw_attachment_id": "raw123",
                    "name": "test.pdf",
                    "email_id": "email123",
                    "processing_status": "downloaded",
                }
            ],
            f,
        )
        path = f.name

    try:
        migrate_manifest(path)
        first_content = open(path).read()
        migrate_manifest(path)
        second_content = open(path).read()
        assert first_content == second_content
    finally:
        os.unlink(path)


def test_refresh_manifest_missing_file():
    attachment = Attachment(
        attachment_id="test123",
        raw_attachment_id="raw123",
        name="test.pdf",
        email_id="email123",
        processing_status="downloaded",
        physical_file="/tmp/nonexistent_file.pdf",
        bank="hsbc",
        instrument="account",
    )

    result = refresh_manifest([attachment])
    assert len(result) == 1
    assert result[0].processing_status == "failed_download"
    assert result[0].metadata["processing_error"] == "Physical file missing"


def test_refresh_clears_parquet_path_for_non_processed():
    attachment = Attachment(
        attachment_id="test123",
        raw_attachment_id="raw123",
        name="test.pdf",
        email_id="email123",
        processing_status="downloaded",
        bank="hsbc",
        instrument="account",
        parquet_path="data/processed/test123.parquet",
    )

    result = refresh_manifest([attachment])
    assert result[0].parquet_path is None


def test_refresh_clears_error_for_processed():
    attachment = Attachment(
        attachment_id="test123",
        raw_attachment_id="raw123",
        name="test.pdf",
        email_id="email123",
        processing_status="processed",
        bank="hsbc",
        instrument="account",
        metadata={"processing_error": "old error"},
    )

    result = refresh_manifest([attachment])
    assert "processing_error" not in result[0].metadata


def test_read_manifest_retry_on_validation_error():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(
            [
                {
                    "attachment_id": "test123",
                    "raw_attachment_id": "raw123",
                    "name": "test.pdf",
                    "email_id": "email123",
                    "processing_status": "downloaded",
                }
            ],
            f,
        )
        path = f.name

    try:
        result = read_manifest(path)
        assert len(result) == 1
        assert result[0].attachment_id == "test123"
        assert result[0].parquet_path is None
    finally:
        os.unlink(path)
