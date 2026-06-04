"""/api/v1/papers/upload*" — single-file and multipart PDF upload."""

from __future__ import annotations

import hashlib
import logging
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.v1._helpers import (
    MAX_MULTIPART_UPLOAD_BYTES,
    MAX_UPLOAD_BYTES,
    enqueue_parse_task,
    ensure_pdf_filename,
    new_task_object_name,
    sanitize_filename,
    validate_md5,
)
from app.core.config import settings
from app.core.schemas import (
    MultipartChunkResponse,
    MultipartCompleteRequest,
    MultipartUploadInitRequest,
    MultipartUploadInitResponse,
    MultipartUploadStatusResponse,
    UploadResponse,
)
from app.services.minio_service import get_minio_service
from app.services.storage.redis import UploadStateError, get_upload_state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF and queue it for async parsing.",
)
async def upload_paper(file: UploadFile = File(...)) -> UploadResponse:
    fname = (file.filename or "").lower()
    accepted_types = {
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
        None,
        "",
    }
    if (file.content_type not in accepted_types) or (not fname.endswith(".pdf")):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF uploads are accepted (got content_type={file.content_type!r}, filename={file.filename!r}).",
        )

    payload = await file.read()
    size = len(payload)
    if size == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File too large ({size} bytes); limit is {MAX_UPLOAD_BYTES}.",
        )

    task_id, safe_name, object_name = new_task_object_name(file.filename or "paper.pdf")
    minio = get_minio_service()
    try:
        await run_in_threadpool(
            minio.upload_bytes,
            object_name,
            payload,
            "application/pdf",
        )
    except Exception as exc:
        logger.exception("MinIO upload failed for task {}: {}", task_id, exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Failed to persist file to object storage.",
        ) from exc

    try:
        response = await enqueue_parse_task(
            task_id=task_id,
            object_name=object_name,
            original_filename=safe_name,
        )
    except Exception as exc:
        logger.exception("Kafka publish failed for task {}: {}", task_id, exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Stored file but failed to enqueue parse task.",
        ) from exc

    logger.info(
        "Upload accepted: task_id={} object={} size={} bytes",
        task_id,
        object_name,
        size,
    )
    return response


@router.post(
    "/upload/multipart/init",
    response_model=MultipartUploadInitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Initialize a reliable multipart PDF upload.",
)
async def init_multipart_upload(
    request: MultipartUploadInitRequest,
) -> MultipartUploadInitResponse:
    safe_name = sanitize_filename(request.filename or "paper.pdf")
    ensure_pdf_filename(safe_name)
    if request.total_size > MAX_MULTIPART_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File too large ({request.total_size} bytes); limit is {MAX_MULTIPART_UPLOAD_BYTES}.",
        )
    if request.file_md5:
        validate_md5(request.file_md5, "file_md5")

    upload_id = str(uuid.uuid4())
    task_id, _, object_name = new_task_object_name(safe_name)
    state = get_upload_state()
    try:
        state.init_upload(
            upload_id=upload_id,
            task_id=task_id,
            filename=safe_name,
            object_name=object_name,
            total_size=request.total_size,
            total_chunks=request.total_chunks,
            file_md5=request.file_md5,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Redis multipart init failed: {}", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Failed to initialize upload state.",
        ) from exc

    return MultipartUploadInitResponse(
        upload_id=upload_id,
        task_id=task_id,
        object_name=object_name,
        uploaded_chunks=[],
        expires_in_seconds=settings.upload_state_ttl_seconds,
    )


@router.get(
    "/upload/multipart/{upload_id}",
    response_model=MultipartUploadStatusResponse,
    summary="Get multipart upload status from Redis Bitmap.",
)
async def get_multipart_upload_status(upload_id: str) -> MultipartUploadStatusResponse:
    try:
        state = get_upload_state()
        meta = state.get_meta(upload_id)
        uploaded_chunks = state.uploaded_chunks(upload_id)
    except UploadStateError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    total_chunks = int(meta["total_chunks"])
    return MultipartUploadStatusResponse(
        upload_id=upload_id,
        task_id=meta["task_id"],
        object_name=meta["object_name"],
        filename=meta["filename"],
        total_size=int(meta["total_size"]),
        total_chunks=total_chunks,
        uploaded_chunks=uploaded_chunks,
        uploaded_count=len(uploaded_chunks),
        complete=len(uploaded_chunks) == total_chunks,
    )


@router.post(
    "/upload/multipart/{upload_id}/chunks",
    response_model=MultipartChunkResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload one PDF chunk with MD5 verification.",
)
async def upload_multipart_chunk(
    upload_id: str,
    chunk_index: int = Form(...),
    chunk_md5: str = Form(...),
    chunk: UploadFile = File(...),
) -> MultipartChunkResponse:
    validate_md5(chunk_md5, "chunk_md5")
    try:
        state = get_upload_state()
        meta = state.get_meta(upload_id)
    except UploadStateError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    total_chunks = int(meta["total_chunks"])
    if chunk_index < 0 or chunk_index >= total_chunks:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "chunk_index out of range.")

    payload = await chunk.read()
    actual_md5 = hashlib.md5(payload).hexdigest()
    if actual_md5.lower() != chunk_md5.lower():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Chunk MD5 mismatch.")

    chunk_object = f"multipart/{upload_id}/chunks/{chunk_index:08d}.part"
    try:
        await run_in_threadpool(
            get_minio_service().upload_bytes,
            chunk_object,
            payload,
            "application/octet-stream",
        )
        uploaded_count = await run_in_threadpool(
            state.mark_chunk,
            upload_id=upload_id,
            chunk_index=chunk_index,
            chunk_md5=chunk_md5,
            chunk_size=len(payload),
            object_name=chunk_object,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Multipart chunk upload failed for {}#{}: {}", upload_id, chunk_index, exc
        )
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Failed to persist uploaded chunk.",
        ) from exc

    return MultipartChunkResponse(
        upload_id=upload_id,
        chunk_index=chunk_index,
        uploaded=True,
        uploaded_count=uploaded_count,
        total_chunks=total_chunks,
    )


@router.post(
    "/upload/multipart/{upload_id}/complete",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Merge verified chunks into MinIO and queue parsing.",
)
async def complete_multipart_upload(
    upload_id: str,
    request: MultipartCompleteRequest,
) -> UploadResponse:
    try:
        state = get_upload_state()
        meta = state.get_meta(upload_id)
        total_chunks = int(meta["total_chunks"])
        uploaded_count = state.uploaded_count(upload_id)
        if uploaded_count != total_chunks:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Upload incomplete: {uploaded_count}/{total_chunks} chunks uploaded.",
            )
        chunk_objects = state.chunk_object_names(upload_id)
    except UploadStateError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    expected_md5 = request.file_md5 or meta.get("file_md5") or None
    if expected_md5:
        validate_md5(expected_md5, "file_md5")

    minio = get_minio_service()
    try:
        await run_in_threadpool(
            minio.merge_objects_to_pdf,
            chunk_object_names=chunk_objects,
            object_name=meta["object_name"],
            expected_size=int(meta["total_size"]),
            expected_md5=expected_md5,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Multipart merge failed for {}: {}", upload_id, exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Failed to merge uploaded chunks.",
        ) from exc

    try:
        response = await enqueue_parse_task(
            task_id=meta["task_id"],
            object_name=meta["object_name"],
            original_filename=meta["filename"],
            message="Multipart upload merged; queued for parsing.",
            upload_id=upload_id,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Kafka publish failed for upload {}: {}", upload_id, exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Merged file but failed to enqueue parse task.",
        ) from exc

    await run_in_threadpool(minio.remove_objects, chunk_objects)
    await run_in_threadpool(state.mark_completed, upload_id)

    return response
