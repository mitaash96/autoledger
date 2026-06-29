"""Manifest filtering and sample-set selection for the autogen pipeline."""

import os

from src import config as cfg
from src.autogen.models import Attachment, SampleSet
from src.email.utils import read_manifest
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])


def filter_attachments(
    attachments: list[Attachment],
    bank: str,
    instrument: str,
) -> list[Attachment]:
    """Return records matching bank+instrument that have an existing physical file."""
    result = [
        a for a in attachments
        if a.bank == bank
        and a.instrument == instrument
        and a.physical_file is not None
        and os.path.exists(a.physical_file)
    ]
    logger.debug(
        "filter_attachments: %d/%d kept (bank=%s, instrument=%s)",
        len(result), len(attachments), bank, instrument,
    )
    return result


def select_samples(attachments: list[Attachment]) -> SampleSet:
    """Pure function: sort by date then split into dev/test with no overlap.

    Algorithm
    ---------
    1. Sort ascending by date; None dates are treated as oldest.
    2. Determine dev count k and test count t from n:
       n=1-2 -> k=n, t=0
       n=3-4 -> k=n-1, t=1
       n=5-6 -> k=n-2, t=2
       n>=7  -> k=5,   t=2
    3. Test  = last t items of sorted list (most recent).
    4. Dev   = uniform-interval sample from prefix sorted[:n-t].
    """
    if not attachments:
        raise ValueError("no attachments available for sampling")

    sorted_attachments = sorted(
        attachments,
        key=lambda a: (a.date is not None, a.date),
    )

    n = len(sorted_attachments)

    if n <= 2:
        k, t = n, 0
    elif n <= 4:
        k, t = n - 1, 1
    elif n <= 6:
        k, t = n - 2, 2
    else:
        k, t = 5, 2

    test = sorted_attachments[n - t:] if t > 0 else []
    prefix = sorted_attachments[: n - t]
    m = len(prefix)

    if k == 1:
        dev_indices = [0]
    else:
        dev_indices = [round(i * (m - 1) / (k - 1)) for i in range(k)]

    dev = [prefix[idx] for idx in dev_indices]

    logger.debug(
        "select_samples: n=%d k=%d t=%d dev_ids=%s test_ids=%s",
        n, k, t,
        [a.attachment_id for a in dev],
        [a.attachment_id for a in test],
    )
    return SampleSet(dev=dev, test=test)


def load_sample_set(
    bank: str,
    instrument: str,
    manifest_path: str = "data/control/email_manifest.json",
) -> SampleSet:
    """Read manifest, filter by bank+instrument, return a SampleSet."""
    attachments = read_manifest(manifest_path)
    filtered = filter_attachments(attachments, bank=bank, instrument=instrument)
    return select_samples(filtered)
