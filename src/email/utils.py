import json
import os
from pathlib import Path

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

from .schemas import Attachment

MANIFEST_PATH = "data/control/email_manifest.json"


def read_manifest(path: str = MANIFEST_PATH, retry: bool = True) -> list[Attachment]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)

    for i, d in enumerate(data):
        try:
            data[i] = Attachment.model_validate(d)
        except ValidationError:
            if retry:
                migrate_manifest(path=path)
                return read_manifest(path, retry=False)
            else:
                raise ValueError(
                    f"Failed to validate attachment: {d.get('attachment_id', 'unknown')}"
                )
    return refresh_manifest(data)


def write_manifest(attachments: list[Attachment], path: str = MANIFEST_PATH):
    attachments = refresh_manifest(attachments)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([a.model_dump(mode="json") for a in attachments], f, indent=2)


def _read_json(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, data: list[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def refresh_manifest(manifest: list[Attachment]) -> list[Attachment]:
    for attachment in manifest:
        if (
            attachment.processing_status != "processed"
            and attachment.physical_file
        ):
            if not os.path.exists(attachment.physical_file):
                attachment.processing_status = "failed_download"
                attachment.metadata["processing_error"] = "Physical file missing"
                attachment.physical_file = None
        elif attachment.processing_status == "processed":
            attachment.metadata.pop("processing_error", None)
        elif attachment.processing_status != "processed" and attachment.parquet_path:
            attachment.parquet_path = None
    return manifest


def migrate_manifest(
    path: str = MANIFEST_PATH,
    model_class: type[BaseModel] = Attachment,
) -> list[Attachment]:
    raw_records = _read_json(path)
    if not raw_records:
        return []

    model_fields = model_class.model_fields
    model_field_names = set(model_fields.keys())

    data_field_names = set()
    for record in raw_records:
        data_field_names.update(record.keys())
    extra_in_data = data_field_names - model_field_names

    migrated = []
    mutations = 0

    for record in raw_records:
        if "processing_error" in record:
            record.setdefault("metadata", {})
            record["metadata"]["processing_error"] = record.pop("processing_error")
            mutations += 1

        for field in extra_in_data - {"processing_error"}:
            if field in record:
                del record[field]
                mutations += 1

        for field, field_info in model_fields.items():
            if field not in record:
                default = field_info.default
                if default is not PydanticUndefined:
                    record[field] = default
                elif field_info.default_factory is not None:
                    record[field] = field_info.default_factory()
                else:
                    record[field] = None
                mutations += 1

        migrated.append(record)

    if mutations > 0:
        _write_json(path, migrated)

    return [Attachment.model_validate(record) for record in migrated]


