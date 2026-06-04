"""Shared helpers for `/api/v1/papers` upload endpoints."""

from __future__ import annotations

import re
import uuid
from datetime import datetime

from fastapi import HTTPException, status

from app.core.schemas import ParseTask, TaskRecord, TaskStatus, UploadResponse
from app.services.storage.kafka import get_kafka_producer
from app.services.minio_service import get_minio_service
from app.services.task_svc import create_task

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_MULTIPART_UPLOAD_BYTES = 512 * 1024 * 1024

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MD5_RE = re.compile(r"^[a-fA-F0-9]{32}$")


def sanitize_filename(name: str) -> str:
    """Strip path separators / unsafe chars while preserving the extension."""
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    cleaned = _SAFE_FILENAME_RE.sub("_", base).strip("._") or "paper.pdf"
    return cleaned[:120]


def ensure_pdf_filename(filename: str) -> None:
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF uploads are accepted."
        )


def validate_md5(value: str, field_name: str) -> None:
    if not _MD5_RE.fullmatch(value):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{field_name} must be a 32-char MD5 hex digest.",
        )


def new_task_object_name(filename: str) -> tuple[str, str, str]:
    """Return (task_id, safe_filename, object_name)."""
    task_id = str(uuid.uuid4())
    safe_name = sanitize_filename(filename or "paper.pdf")
    object_name = f"papers/{task_id}/{safe_name}"
    return task_id, safe_name, object_name


async def enqueue_parse_task(
    *,
    task_id: str,
    object_name: str,
    original_filename: str,
    message: str = "Uploaded; queued for parsing.",
    upload_id: str | None = None,
) -> UploadResponse:
    minio = get_minio_service()
    create_task(
        TaskRecord(
            task_id=task_id,
            object_name=object_name,
            original_filename=original_filename,
            status=TaskStatus.PENDING,
            message=message,
            upload_id=upload_id,
        )
    )
    task = ParseTask(
        task_id=task_id,
        object_name=object_name,
        bucket=minio.bucket,
        original_filename=original_filename,
        submitted_at=datetime.utcnow(),
    )
    await get_kafka_producer().send_parse_task(task)
    return UploadResponse(
        task_id=task_id,
        object_name=object_name,
        status=TaskStatus.PENDING,
        message=message,
    )
