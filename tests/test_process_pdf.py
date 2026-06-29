from unittest.mock import patch, MagicMock

import json
import os
import tempfile

import pikepdf
import polars as pl
import pytest

from src.email.process_pdf import _discover_pipelines, _get_pipeline, _process_one, main
from src.email.schemas import Attachment


def test_discover_pipelines_finds_hsbc_account():
    registry = _discover_pipelines()
    assert ("hsbc", "account") in registry
    assert registry[("hsbc", "account")] == "src.email.pipelines.hsbc_account"


def test_get_pipeline_raises_for_unknown():
    with pytest.raises(ValueError, match="No pipeline registered"):
        _get_pipeline("unknown_bank", "account")


def test_get_pipeline_returns_module():
    module = _get_pipeline("hsbc", "account")
    assert hasattr(module, "transform")


def test_process_one_success():
    attachment = Attachment(
        attachment_id="test123",
        raw_attachment_id="raw123",
        name="test.pdf",
        email_id="email123",
        processing_status="downloaded",
        physical_file="data/pdf/test.pdf",
        bank="hsbc",
        instrument="account",
    )

    mock_df = pl.DataFrame({"col": [1, 2, 3]})

    with (
        patch("src.email.process_pdf._get_pipeline") as mock_get,
        patch.object(mock_df, "write_parquet") as mock_write,
    ):
        mock_module = MagicMock()
        mock_module.transform.return_value = mock_df
        mock_get.return_value = mock_module

        _process_one(attachment, None)
        assert attachment.parquet_path == "data/processed/test123.parquet"
        mock_write.assert_called_once_with("data/processed/test123.parquet")


def test_process_one_password_error():
    attachment = Attachment(
        attachment_id="test123",
        raw_attachment_id="raw123",
        name="test.pdf",
        email_id="email123",
        processing_status="downloaded",
        physical_file="data/pdf/test.pdf",
        bank="hsbc",
        instrument="account",
    )

    with patch("src.email.process_pdf._get_pipeline") as mock_get:
        mock_module = MagicMock()
        mock_module.transform.side_effect = pikepdf.PasswordError("wrong password")
        mock_get.return_value = mock_module

        with pytest.raises(pikepdf.PasswordError):
            _process_one(attachment, None)


def test_process_one_generic_failure():
    attachment = Attachment(
        attachment_id="test123",
        raw_attachment_id="raw123",
        name="test.pdf",
        email_id="email123",
        processing_status="downloaded",
        physical_file="data/pdf/test.pdf",
        bank="hsbc",
        instrument="account",
    )

    with patch("src.email.process_pdf._get_pipeline") as mock_get:
        mock_module = MagicMock()
        mock_module.transform.side_effect = RuntimeError("corrupted pdf")
        mock_get.return_value = mock_module

        with pytest.raises(RuntimeError):
            _process_one(attachment, None)


def test_process_one_null_dataframe():
    attachment = Attachment(
        attachment_id="test123",
        raw_attachment_id="raw123",
        name="test.pdf",
        email_id="email123",
        processing_status="downloaded",
        physical_file="data/pdf/test.pdf",
        bank="hsbc",
        instrument="account",
    )

    with patch("src.email.process_pdf._get_pipeline") as mock_get:
        mock_module = MagicMock()
        mock_module.transform.return_value = None
        mock_get.return_value = mock_module

        with pytest.raises(ValueError, match="dataframe is None"):
            _process_one(attachment, None)


def test_main_sets_status_and_path():
    attachment = Attachment(
        attachment_id="test123",
        raw_attachment_id="raw123",
        name="test.pdf",
        email_id="email123",
        processing_status="downloaded",
        physical_file="data/pdf/test.pdf",
        bank="hsbc",
        instrument="account",
    )

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump([attachment.model_dump(mode="json")], f)
        manifest_path = f.name

    try:
        from concurrent.futures import Future

        success_future = Future()
        success_future.set_result(None)

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.return_value = success_future

        with (
            patch("src.email.process_pdf.MANIFEST_PATH", manifest_path),
            patch(
                "src.email.process_pdf.ProcessPoolExecutor", return_value=mock_executor
            ),
            patch("src.email.process_pdf.write_manifest") as mock_write,
            patch("src.email.utils.refresh_manifest", side_effect=lambda x: x),
        ):
            result = main(mode="full", max_workers=1)

            processed = [a for a in result if a.attachment_id == "test123"][0]
            assert processed.processing_status == "processed"
            assert processed.parquet_path == "data/processed/test123.parquet"
            mock_write.assert_called_once()
    finally:
        os.unlink(manifest_path)
