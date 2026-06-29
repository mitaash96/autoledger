"""Tests for src.autogen.extraction.runner (Task 6).

Strategy
--------
ProcessPoolExecutor cannot propagate mocks/patches into subprocesses.  All
executor-level patching therefore substitutes a synchronous *DummyExecutor*
that calls fn(*args) in-process and wraps the result/exception in a
future-like object.  extract_one is separately patched to return canned data,
so real PDF extraction is never triggered.  The textlayer reference runs
inline (no executor), but also calls the patched extract_one.
"""

from __future__ import annotations

import concurrent.futures
from unittest import mock

import pytest

from src.autogen.extraction.runner import (
    EXTRACTOR_CLASSES,
    LIBRARY_NAMES,
    REFERENCE_NAME,
    extract_one,
    run_all_extractors,
)
from src.autogen.models import ExtractedTable


# ---------------------------------------------------------------------------
# Helpers shared by tests
# ---------------------------------------------------------------------------

ALL_NAMES = [REFERENCE_NAME] + LIBRARY_NAMES  # ["textlayer", "docling", "camelot", "pymupdf", "pdfplumber"]


def _make_table(label: str) -> ExtractedTable:
    return ExtractedTable(name=label, rows=[[label]], page=1)


CANNED: dict[str, list[ExtractedTable]] = {name: [_make_table(name)] for name in ALL_NAMES}


def _fake_extract_one(name: str, physical_file: str, password: str) -> list[ExtractedTable]:
    return CANNED[name]


# ---------------------------------------------------------------------------
# Synchronous dummy executors
# ---------------------------------------------------------------------------


class _DummyFuture:
    """Runs fn immediately; .result() returns the value or re-raises."""

    def __init__(self, fn, *args):
        try:
            self._value = fn(*args)
            self._exc = None
        except Exception as exc:  # noqa: BLE001
            self._value = None
            self._exc = exc

    def result(self, timeout=None):
        if self._exc is not None:
            raise self._exc
        return self._value


class _DummyExecutor:
    """Context-manager executor that runs fn(*args) synchronously."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def submit(self, fn, *args):
        return _DummyFuture(fn, *args)


def _dummy_executor_factory(*args, **kwargs):
    """Drop-in replacement for ProcessPoolExecutor."""
    return _DummyExecutor()


# ---------------------------------------------------------------------------
# extract_one tests
# ---------------------------------------------------------------------------


class TestExtractOne:
    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown extractor"):
            extract_one("nonexistent", "/some/file.pdf", "")

    def test_known_name_instantiates_correct_class_and_returns_output(self):
        sentinel_tables = [_make_table("sentinel")]
        mock_instance = mock.MagicMock()
        mock_instance.extract.return_value = sentinel_tables
        mock_cls = mock.MagicMock(return_value=mock_instance)

        with mock.patch.dict(
            "src.autogen.extraction.runner.EXTRACTOR_CLASSES",
            {"docling": mock_cls},
        ):
            result = extract_one("docling", "/fake.pdf", "secret")

        mock_cls.assert_called_once_with()
        mock_instance.extract.assert_called_once_with("/fake.pdf", "secret")
        # extract_one rebuilds tables through promote_header; content is preserved.
        assert result == sentinel_tables

    @pytest.mark.parametrize("name", list(EXTRACTOR_CLASSES))
    def test_all_registry_names_are_accepted(self, name: str):
        """Each name in EXTRACTOR_CLASSES must not raise ValueError."""
        mock_instance = mock.MagicMock()
        mock_instance.extract.return_value = []
        mock_cls = mock.MagicMock(return_value=mock_instance)

        with mock.patch.dict(
            "src.autogen.extraction.runner.EXTRACTOR_CLASSES",
            {name: mock_cls},
        ):
            result = extract_one(name, "/fake.pdf", "")

        assert result == []


# ---------------------------------------------------------------------------
# run_all_extractors tests
# ---------------------------------------------------------------------------


class TestRunAllExtractors:
    """Patches: ProcessPoolExecutor, extract_one, cache."""

    _PATCH_PROC = "src.autogen.extraction.runner.ProcessPoolExecutor"
    _PATCH_EXTRACT = "src.autogen.extraction.runner.extract_one"
    _PATCH_CACHE = "src.autogen.extraction.runner.cache"

    # ---- helpers -----------------------------------------------------------

    def _run(
        self,
        mock_extract,
        mock_cache,
        cache_hits: dict[str, list[ExtractedTable]] | None = None,
        use_cache: bool = True,
        timeout_seconds: int = 120,
    ) -> dict[str, list[ExtractedTable]]:
        """Wire up executor and delegate to run_all_extractors."""
        cache_hits = cache_hits or {}

        def cache_load(attachment_id, name):
            return cache_hits.get(name)

        mock_cache.load.side_effect = cache_load
        mock_cache.save.return_value = None

        with mock.patch(self._PATCH_PROC, side_effect=_dummy_executor_factory):
            return run_all_extractors(
                attachment_id="att_001",
                physical_file="/fake.pdf",
                password="pw",
                timeout_seconds=timeout_seconds,
                use_cache=use_cache,
            )

    # ---- tests -------------------------------------------------------------

    def test_all_uncached_returns_all_five_names(self):
        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one) as mock_extract,
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            result = self._run(mock_extract, mock_cache, cache_hits={})

        assert set(result.keys()) == set(ALL_NAMES)

    def test_all_uncached_calls_extract_once_per_name(self):
        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one) as mock_extract,
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            self._run(mock_extract, mock_cache, cache_hits={})

        called_names = [call.args[0] for call in mock_extract.call_args_list]
        assert sorted(called_names) == sorted(ALL_NAMES)

    def test_all_uncached_calls_cache_save_for_each_name(self):
        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one) as mock_extract,
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            self._run(mock_extract, mock_cache, cache_hits={})

        saved_names = [call.args[1] for call in mock_cache.save.call_args_list]
        assert sorted(saved_names) == sorted(ALL_NAMES)

    def test_all_uncached_returns_correct_tables(self):
        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one) as mock_extract,
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            result = self._run(mock_extract, mock_cache, cache_hits={})

        for name in ALL_NAMES:
            assert result[name] == CANNED[name]

    # ---- partial cache hit ------------------------------------------------

    def test_cache_hit_extractors_not_re_run(self):
        cached_names = ["docling", "pymupdf"]
        cache_hits = {name: CANNED[name] for name in cached_names}

        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one) as mock_extract,
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            result = self._run(mock_extract, mock_cache, cache_hits=cache_hits)

        called_names = [call.args[0] for call in mock_extract.call_args_list]
        for name in cached_names:
            assert name not in called_names

    def test_cache_hit_extractors_in_result(self):
        cached_names = ["docling", "pymupdf"]
        cache_hits = {name: CANNED[name] for name in cached_names}

        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one) as mock_extract,
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            result = self._run(mock_extract, mock_cache, cache_hits=cache_hits)

        for name in cached_names:
            assert result[name] == CANNED[name]

    def test_cache_hit_result_has_all_five_names(self):
        cache_hits = {"docling": CANNED["docling"]}

        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one) as mock_extract,
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            result = self._run(mock_extract, mock_cache, cache_hits=cache_hits)

        assert set(result.keys()) == set(ALL_NAMES)

    # ---- all cached -------------------------------------------------------

    def test_all_cached_extract_one_never_called(self):
        cache_hits = {name: CANNED[name] for name in ALL_NAMES}

        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one) as mock_extract,
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            self._run(mock_extract, mock_cache, cache_hits=cache_hits)

        mock_extract.assert_not_called()

    def test_all_cached_result_has_all_five_names(self):
        cache_hits = {name: CANNED[name] for name in ALL_NAMES}

        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one) as mock_extract,
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            result = self._run(mock_extract, mock_cache, cache_hits=cache_hits)

        assert set(result.keys()) == set(ALL_NAMES)

    def test_all_cached_returns_correct_tables(self):
        cache_hits = {name: CANNED[name] for name in ALL_NAMES}

        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one) as mock_extract,
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            result = self._run(mock_extract, mock_cache, cache_hits=cache_hits)

        for name in ALL_NAMES:
            assert result[name] == CANNED[name]

    # ---- timeout / exception (library extractor via ProcessPool) ----------

    def test_timeout_extractor_maps_to_empty_list(self):
        failing_name = "camelot"

        def fake_extract_with_timeout(name, physical_file, password):
            if name == failing_name:
                raise concurrent.futures.TimeoutError()
            return CANNED[name]

        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=fake_extract_with_timeout),
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            result = self._run(None, mock_cache, cache_hits={})

        assert result[failing_name] == []

    def test_timeout_extractor_not_saved_to_cache(self):
        failing_name = "camelot"

        def fake_extract_with_timeout(name, physical_file, password):
            if name == failing_name:
                raise concurrent.futures.TimeoutError()
            return CANNED[name]

        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=fake_extract_with_timeout),
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            self._run(None, mock_cache, cache_hits={})

        saved_names = [call.args[1] for call in mock_cache.save.call_args_list]
        assert failing_name not in saved_names

    def test_timeout_other_extractors_unaffected(self):
        failing_name = "camelot"

        def fake_extract_with_timeout(name, physical_file, password):
            if name == failing_name:
                raise concurrent.futures.TimeoutError()
            return CANNED[name]

        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=fake_extract_with_timeout),
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            result = self._run(None, mock_cache, cache_hits={})

        for name in ALL_NAMES:
            if name != failing_name:
                assert result[name] == CANNED[name]

    def test_exception_extractor_maps_to_empty_list(self):
        failing_name = "pdfplumber"

        def fake_extract_with_exc(name, physical_file, password):
            if name == failing_name:
                raise RuntimeError("unexpected error")
            return CANNED[name]

        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=fake_extract_with_exc),
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            result = self._run(None, mock_cache, cache_hits={})

        assert result[failing_name] == []

    def test_exception_extractor_not_saved_to_cache(self):
        failing_name = "pdfplumber"

        def fake_extract_with_exc(name, physical_file, password):
            if name == failing_name:
                raise RuntimeError("unexpected error")
            return CANNED[name]

        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=fake_extract_with_exc),
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            self._run(None, mock_cache, cache_hits={})

        saved_names = [call.args[1] for call in mock_cache.save.call_args_list]
        assert failing_name not in saved_names

    # ---- inline reference exception ---------------------------------------

    def test_inline_reference_exception_maps_to_empty_list(self):
        """An exception in the inline textlayer path must yield [] and not be cached."""

        def fake_extract_ref_fails(name, physical_file, password):
            if name == REFERENCE_NAME:
                raise RuntimeError("text layer unreadable")
            return CANNED[name]

        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=fake_extract_ref_fails),
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            result = self._run(None, mock_cache, cache_hits={})

        assert result[REFERENCE_NAME] == []
        saved_names = [call.args[1] for call in mock_cache.save.call_args_list]
        assert REFERENCE_NAME not in saved_names

    # ---- use_cache=False --------------------------------------------------

    def test_no_cache_load_never_called(self):
        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one),
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            self._run(None, mock_cache, cache_hits={}, use_cache=False)

        mock_cache.load.assert_not_called()

    def test_no_cache_save_never_called(self):
        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one),
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            self._run(None, mock_cache, cache_hits={}, use_cache=False)

        mock_cache.save.assert_not_called()

    def test_no_cache_returns_all_five_names(self):
        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one),
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            result = self._run(None, mock_cache, cache_hits={}, use_cache=False)

        assert set(result.keys()) == set(ALL_NAMES)

    def test_no_cache_still_runs_all_extractors(self):
        with (
            mock.patch(self._PATCH_EXTRACT, side_effect=_fake_extract_one) as mock_extract,
            mock.patch(self._PATCH_CACHE) as mock_cache,
        ):
            self._run(None, mock_cache, cache_hits={}, use_cache=False)

        called_names = [call.args[0] for call in mock_extract.call_args_list]
        assert sorted(called_names) == sorted(ALL_NAMES)


class TestPromoteHeader:
    """promote_header strips leading title/junk so rows[0] is the real header."""

    def test_header_already_first_is_unchanged(self):
        from src.autogen.extraction.base import promote_header

        rows = [
            ["Txn Date", "Narration", "Withdrawals", "Closing Balance"],
            ["22/03/2018", "XOR 161849", "1054.71", "18,785.46"],
        ]
        assert promote_header(rows) == rows

    def test_title_above_header_is_dropped(self):
        from src.autogen.extraction.base import promote_header

        rows = [
            ["Statement of Transactions in Savings Account XXXX", "", "", ""],
            ["DATE", "PARTICULARS", "WITHDRAWALS(INR)", "BALANCE(₹)"],
            ["24-05-2019", "B/F", "527.89", "31,461.97"],
        ]
        out = promote_header(rows)
        assert out[0] == ["DATE", "PARTICULARS", "WITHDRAWALS(INR)", "BALANCE(₹)"]
        assert len(out) == 2

    def test_decorated_header_cells_survive(self):
        from src.autogen.extraction.base import _is_data_cell

        for label in ("WITHDRAWALS(INR)", "BALANCE(₹)", "Deposit Amt(Cr)", "DATE"):
            assert not _is_data_cell(label)

    def test_filler_row_between_header_and_data(self):
        from src.autogen.extraction.base import promote_header

        rows = [
            ["Statement of Transactions in Savings Account XXXX", "", "", ""],
            ["DATE", "PARTICULARS", "WITHDRAWALS", "BALANCE"],
            ["", "I/", "", ""],  # stray opening-balance fragment
            ["05-06-2019", "B/F", "527.89", "31,461.97"],
        ]
        out = promote_header(rows)
        assert out[0] == ["DATE", "PARTICULARS", "WITHDRAWALS", "BALANCE"]
        assert len(out) == 3

    def test_no_data_row_unchanged(self):
        from src.autogen.extraction.base import promote_header

        rows = [["Header A", "Header B"], ["only", "labels"]]
        assert promote_header(rows) == rows
