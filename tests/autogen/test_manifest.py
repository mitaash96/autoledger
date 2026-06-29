"""Tests for src.autogen.manifest (TDD)."""

import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.autogen.manifest import filter_attachments, load_sample_set, select_samples
from src.autogen.models import Attachment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_attachment(
    attachment_id: str,
    bank: str | None = "icici",
    instrument: str | None = "savings",
    physical_file: str | None = None,
    date: datetime.datetime | None = None,
) -> Attachment:
    return Attachment(
        attachment_id=attachment_id,
        raw_attachment_id=attachment_id,
        name=f"{attachment_id}.pdf",
        email_id=f"email_{attachment_id}",
        bank=bank,
        instrument=instrument,
        physical_file=physical_file,
        date=date,
    )


def dt(year: int, month: int = 1, day: int = 1) -> datetime.datetime:
    return datetime.datetime(year, month, day)


# ---------------------------------------------------------------------------
# filter_attachments
# ---------------------------------------------------------------------------


class TestFilterAttachments:
    def test_keeps_matching_with_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "a.pdf"
        f.write_text("x")
        a = make_attachment("a1", bank="icici", instrument="savings", physical_file=str(f))
        result = filter_attachments([a], bank="icici", instrument="savings")
        assert result == [a]

    def test_drops_wrong_bank(self, tmp_path: Path) -> None:
        f = tmp_path / "b.pdf"
        f.write_text("x")
        a = make_attachment("b1", bank="hdfc", instrument="savings", physical_file=str(f))
        result = filter_attachments([a], bank="icici", instrument="savings")
        assert result == []

    def test_drops_wrong_instrument(self, tmp_path: Path) -> None:
        f = tmp_path / "c.pdf"
        f.write_text("x")
        a = make_attachment("c1", bank="icici", instrument="credit", physical_file=str(f))
        result = filter_attachments([a], bank="icici", instrument="savings")
        assert result == []

    def test_drops_none_physical_file(self) -> None:
        a = make_attachment("d1", bank="icici", instrument="savings", physical_file=None)
        result = filter_attachments([a], bank="icici", instrument="savings")
        assert result == []

    def test_drops_nonexistent_physical_file(self) -> None:
        a = make_attachment(
            "e1", bank="icici", instrument="savings",
            physical_file="/nonexistent/path/to/file.pdf"
        )
        result = filter_attachments([a], bank="icici", instrument="savings")
        assert result == []

    def test_mixed_list(self, tmp_path: Path) -> None:
        good_f = tmp_path / "good.pdf"
        good_f.write_text("x")
        good = make_attachment("g1", bank="icici", instrument="savings", physical_file=str(good_f))
        wrong_bank = make_attachment("g2", bank="sbi", instrument="savings", physical_file=str(good_f))
        wrong_inst = make_attachment("g3", bank="icici", instrument="credit", physical_file=str(good_f))
        no_file = make_attachment("g4", bank="icici", instrument="savings", physical_file=None)
        missing = make_attachment("g5", bank="icici", instrument="savings", physical_file="/nope.pdf")

        result = filter_attachments(
            [good, wrong_bank, wrong_inst, no_file, missing],
            bank="icici", instrument="savings",
        )
        assert result == [good]


# ---------------------------------------------------------------------------
# select_samples
# ---------------------------------------------------------------------------


class TestSelectSamples:
    def _make_n(self, n: int, start_year: int = 2020) -> list[Attachment]:
        """Create n attachments with sequential dates."""
        return [
            make_attachment(f"id{i}", date=dt(start_year + i))
            for i in range(n)
        ]

    def _ids(self, attachments: list[Attachment]) -> set[str]:
        return {a.attachment_id for a in attachments}

    def test_n0_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="no attachments available for sampling"):
            select_samples([])

    def test_n1_dev1_test0(self) -> None:
        items = self._make_n(1)
        ss = select_samples(items)
        assert len(ss.dev) == 1
        assert len(ss.test) == 0
        assert self._ids(ss.dev).isdisjoint(self._ids(ss.test))

    def test_n2_dev2_test0(self) -> None:
        items = self._make_n(2)
        ss = select_samples(items)
        assert len(ss.dev) == 2
        assert len(ss.test) == 0
        assert self._ids(ss.dev).isdisjoint(self._ids(ss.test))

    def test_n3_dev2_test1(self) -> None:
        items = self._make_n(3)
        ss = select_samples(items)
        assert len(ss.dev) == 2
        assert len(ss.test) == 1
        assert self._ids(ss.dev).isdisjoint(self._ids(ss.test))

    def test_n4_dev3_test1(self) -> None:
        items = self._make_n(4)
        ss = select_samples(items)
        assert len(ss.dev) == 3
        assert len(ss.test) == 1
        assert self._ids(ss.dev).isdisjoint(self._ids(ss.test))

    def test_n5_dev3_test2(self) -> None:
        items = self._make_n(5)
        ss = select_samples(items)
        assert len(ss.dev) == 3
        assert len(ss.test) == 2
        assert self._ids(ss.dev).isdisjoint(self._ids(ss.test))

    def test_n6_dev4_test2(self) -> None:
        items = self._make_n(6)
        ss = select_samples(items)
        assert len(ss.dev) == 4
        assert len(ss.test) == 2
        assert self._ids(ss.dev).isdisjoint(self._ids(ss.test))

    def test_n7_dev5_test2(self) -> None:
        items = self._make_n(7)
        ss = select_samples(items)
        assert len(ss.dev) == 5
        assert len(ss.test) == 2
        assert self._ids(ss.dev).isdisjoint(self._ids(ss.test))

    def test_n9_dev5_test2(self) -> None:
        items = self._make_n(9)
        ss = select_samples(items)
        assert len(ss.dev) == 5
        assert len(ss.test) == 2
        assert self._ids(ss.dev).isdisjoint(self._ids(ss.test))

    def test_test_set_is_most_recent(self) -> None:
        """Test set must be the t most-recent items (by date)."""
        items = self._make_n(5)  # dates 2020..2024; n=5 -> t=2
        ss = select_samples(items)
        # sorted ascending: id0(2020) id1(2021) id2(2022) id3(2023) id4(2024)
        # test = last 2 = id3, id4
        test_ids = self._ids(ss.test)
        assert "id3" in test_ids
        assert "id4" in test_ids

    def test_none_dates_sort_oldest(self) -> None:
        """None dates treated as oldest (appear at start of sorted list)."""
        no_date = make_attachment("no_date", date=None)
        old = make_attachment("old", date=dt(2020))
        recent = make_attachment("recent", date=dt(2024))
        # n=3 -> dev=2, test=1; test = most recent = 'recent'
        ss = select_samples([recent, no_date, old])
        assert "recent" in self._ids(ss.test)
        # no_date should NOT be in test
        assert "no_date" not in self._ids(ss.test)

    def test_disjoint_for_all_n(self) -> None:
        """Exhaustive disjoint check for n=1..10."""
        for n in range(1, 11):
            items = self._make_n(n)
            ss = select_samples(items)
            assert self._ids(ss.dev).isdisjoint(self._ids(ss.test)), f"overlap at n={n}"


# ---------------------------------------------------------------------------
# load_sample_set
# ---------------------------------------------------------------------------


class TestLoadSampleSet:
    def test_composes_read_filter_select(self, tmp_path: Path) -> None:
        """load_sample_set should call read_manifest, filter, then select_samples."""
        good_f = tmp_path / "good.pdf"
        good_f.write_text("x")

        good = make_attachment(
            "good1", bank="icici", instrument="savings",
            physical_file=str(good_f), date=dt(2022)
        )
        bad_bank = make_attachment(
            "bad_bank", bank="sbi", instrument="savings",
            physical_file=str(good_f), date=dt(2023)
        )

        with patch("src.autogen.manifest.read_manifest", return_value=[good, bad_bank]):
            ss = load_sample_set(
                bank="icici",
                instrument="savings",
                manifest_path="fake/path.json",
            )

        # Only 'good' passed filtering -> n=1 -> dev=1, test=0
        assert len(ss.dev) == 1
        assert ss.dev[0].attachment_id == "good1"
        assert len(ss.test) == 0

    def test_passes_manifest_path_to_read_manifest(self, tmp_path: Path) -> None:
        f = tmp_path / "f.pdf"
        f.write_text("x")
        a = make_attachment("x1", bank="hdfc", instrument="credit", physical_file=str(f), date=dt(2023))

        with patch("src.autogen.manifest.read_manifest", return_value=[a]) as mock_rm:
            load_sample_set(bank="hdfc", instrument="credit", manifest_path="custom/path.json")
            mock_rm.assert_called_once_with("custom/path.json")

    def test_raises_when_no_attachments_match_filter(self) -> None:
        """When filtering leaves zero attachments, select_samples' ValueError propagates."""
        a = make_attachment("x1", bank="sbi", instrument="savings", physical_file=None, date=dt(2023))

        with patch("src.autogen.manifest.read_manifest", return_value=[a]):
            with pytest.raises(ValueError):
                load_sample_set(bank="icici", instrument="savings", manifest_path="fake.json")
