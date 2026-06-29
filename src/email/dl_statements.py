import base64
import datetime
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from googleapiclient.discovery import Resource

from .. import config as cfg
from ..logger import get_logger
from .gmail_service import get_gmail_service
from .schemas import Attachment
from .utils import (
    MANIFEST_PATH,
    read_manifest,
    write_manifest,
)

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

BANK_ACCOUNTS = cfg.banks
MAX_THREADS = cfg.concurrency_model["email"]["max_workers"]
OUTPUT_DIR = cfg.io_dir["email"]["output"]

JUNK_PDFS = frozenset({"Most Important Terms & Conditions.pdf", "Important_Notes.pdf"})


def form_mail_query(lookback_years: int = 1) -> str:
    date_to = datetime.datetime.now()
    date_from = date_to - datetime.timedelta(days=365 * lookback_years)

    mail_query = f"{cfg.mail_query} AND after:{date_from.strftime('%Y-%m-%d')} before:{date_to.strftime('%Y-%m-%d')}"
    return mail_query


def find_substr(text: str, substrings: list[str]) -> str | None:
    for substring in substrings:
        if substring.lower() in text.lower():
            return substring
    return None


def fetch_message(message_id):
    """Fetch a single message by ID with dedicated service instance."""
    service: Resource = get_gmail_service()
    return (
        service.users()  # type:ignore
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )


def parse_message(m) -> list[Attachment]:
    """Parse message into a list of Attachment models."""
    date = datetime.datetime.fromtimestamp(int(m["internalDate"]) / 1000)
    headers = {h["name"].lower(): h["value"] for h in m["payload"].get("headers", [])}

    return [
        Attachment(
            attachment_id=hashlib.shake_256(
                f"{m['id']}{part['filename']}".encode()
            ).hexdigest(8),
            raw_attachment_id=part["body"]["attachmentId"],
            name=part["filename"],
            email_id=m["id"],
            date=date,
            sender=headers.get("from"),
            email_subject=headers.get("subject"),
        )
        for part in m["payload"].get("parts", [])
        if part.get("filename", "").lower().endswith(".pdf")
    ]


def query_mailbox(lookback_years: int = 1) -> list[dict]:

    service: Resource = get_gmail_service()
    query = form_mail_query(lookback_years)

    messages: list[dict] = []
    page_token: str | None = None
    while True:
        request = (
            service.users()  # type:ignore
            .messages()
            .list(
                userId="me",
                labelIds=["INBOX"],
                q=query,
                maxResults=500,
            )
        )
        if page_token:
            request.pageToken = page_token

        results = request.execute()
        messages.extend(results.get("messages", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break

    logger.info(f"messages returned via query: {len(messages)}")

    return messages


def generate_mail_details(max_workers: int = MAX_THREADS, **kwargs) -> list[Attachment]:

    if "lookback_years" in kwargs:
        messages = query_mailbox(lookback_years=kwargs["lookback_years"])
    else:
        messages = query_mailbox(lookback_years=1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_message, m["id"]): m for m in messages}
        fetched = [future.result() for future in as_completed(futures)]

    return [a for msg in fetched for a in parse_message(msg)]


def enrich_attachment(attachment: Attachment, bank_accounts: list[str]) -> Attachment:
    """Compute derived fields (bank, instrument) on an attachment."""
    subject_lower = (attachment.email_subject or "").lower()
    attachment.instrument = "card" if "card" in subject_lower else "account"
    attachment.bank = find_substr(
        attachment.email_subject or "", bank_accounts
    ) or find_substr(attachment.sender or "", bank_accounts)
    return attachment


def download_attachment(
    attachment: Attachment,
    output_dir: str = OUTPUT_DIR,
) -> Attachment:
    """Download a single attachment. Updates physical_file and processing_status."""
    service: Resource = get_gmail_service()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    api_attachment = (
        service.users()  # type:ignore
        .messages()
        .attachments()
        .get(
            userId="me",
            messageId=attachment.email_id,
            id=attachment.raw_attachment_id,
        )
        .execute()
    )

    file_data = base64.urlsafe_b64decode(api_attachment["data"])
    filepath = os.path.join(output_dir, f"{attachment.attachment_id}_{attachment.name}")

    with open(filepath, "wb") as f:
        f.write(file_data)

    attachment.physical_file = filepath
    attachment.processing_status = "downloaded"
    attachment.metadata["processing_timestamp"] = datetime.datetime.now().isoformat()
    logger.info(f"downloaded attachment {attachment.attachment_id}")
    return attachment


def main(mode: str = "append", lookback_years: int = 1) -> bool | None:

    if not os.path.exists(MANIFEST_PATH):
        mode = "full"

    if mode == "full":
        lookback_years = 10

    all_attachments = generate_mail_details(lookback_years=lookback_years)

    if not all_attachments:
        logger.info("no new emails found")
        return None

    # Filter junk PDFs
    all_attachments = [a for a in all_attachments if a.name not in JUNK_PDFS]

    # In append mode, refresh manifest and dedup
    manifest: list[Attachment] = []
    if mode == "append":
        manifest = read_manifest()
        skip_statuses = frozenset(["downloaded", "processed", "failed_processing"])
        existing = [
            a.attachment_id for a in manifest if a.processing_status in skip_statuses
        ]
        all_attachments = [
            a for a in all_attachments if a.attachment_id not in existing
        ]

    if not all_attachments:
        logger.info("no new attachments to process")
        return None

    # Enrich derived fields
    for a in all_attachments:
        enrich_attachment(a, BANK_ACCOUNTS)

    # Download concurrently
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {
            executor.submit(download_attachment, a, OUTPUT_DIR): a
            for a in all_attachments
        }
        results: list[Attachment] = []
        for future in as_completed(futures):
            original = futures[future]
            try:
                updated = future.result()
                results.append(updated)
            except Exception as e:
                original.processing_status = "failed_download"
                original.metadata["error_details"] = str(e)
                results.append(original)
                logger.error(f"[download] ERROR {original.name}: {e}")

    if not any(a.processing_status == "downloaded" for a in results):
        logger.warning("no attachments were downloaded successfully")
        return None

    # Merge with existing manifest
    if mode == "append":
        result_ids = {a.attachment_id for a in results}
        manifest = [a for a in manifest if a.attachment_id not in result_ids]
        manifest.extend(results)
    else:
        manifest = results

    write_manifest(manifest)
    return True


if __name__ == "__main__":
    main(mode="append")
