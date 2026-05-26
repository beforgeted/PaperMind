"""MinIO object-storage helper.

Wraps the synchronous `minio` SDK behind small, intention-revealing methods.
The wrapper is used by:
    - the FastAPI upload endpoint (uploads PDFs)
    - the Kafka consumer worker (downloads PDFs to parse)

All MinIO calls are sync; FastAPI handlers offload them to a threadpool via
`run_in_threadpool` so we never block the event loop.
"""
from __future__ import annotations

import hashlib
import tempfile
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Optional, Sequence

from minio import Minio
from minio.error import S3Error

from app.core.config import settings
from app.core.logging import logger


class MinIOService:
    """Thin singleton wrapper around `minio.Minio`."""

    def __init__(self) -> None:
        self._client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self._bucket = settings.minio_bucket
        self._bucket_ready = False

    # ---------- bucket ----------

    def ensure_bucket(self) -> None:
        """Idempotent: create the configured bucket if it doesn't exist."""
        if self._bucket_ready:
            return
        try:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
                logger.info("Created MinIO bucket '{}'", self._bucket)
            self._bucket_ready = True
        except S3Error as exc:
            logger.exception("Failed to ensure bucket '{}': {}", self._bucket, exc)
            raise

    # ---------- upload ----------

    def upload_fileobj(
        self,
        object_name: str,
        data: BinaryIO,
        length: int,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a binary stream and return the object key."""
        self.ensure_bucket()
        try:
            self._client.put_object(
                bucket_name=self._bucket,
                object_name=object_name,
                data=data,
                length=length,
                content_type=content_type,
            )
            logger.info("Uploaded {} ({} bytes) to MinIO", object_name, length)
            return object_name
        except S3Error as exc:
            logger.exception("MinIO put_object failed for {}: {}", object_name, exc)
            raise

    def upload_bytes(
        self,
        object_name: str,
        payload: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        return self.upload_fileobj(
            object_name=object_name,
            data=BytesIO(payload),
            length=len(payload),
            content_type=content_type,
        )

    # ---------- download ----------

    def download_to_path(self, object_name: str, dest: str | Path) -> Path:
        """Stream an object to a local file (used by the Docling worker)."""
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.fget_object(self._bucket, object_name, str(dest_path))
            return dest_path
        except S3Error as exc:
            logger.exception("MinIO fget_object failed for {}: {}", object_name, exc)
            raise

    def get_object_bytes(self, object_name: str) -> bytes:
        response = self._client.get_object(self._bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def merge_objects_to_pdf(
        self,
        *,
        chunk_object_names: Sequence[str],
        object_name: str,
        expected_size: int,
        expected_md5: Optional[str] = None,
    ) -> str:
        """Merge MinIO chunk objects into one PDF and verify size / optional MD5."""
        self.ensure_bucket()
        hasher = hashlib.md5()
        total = 0
        with tempfile.TemporaryFile() as merged:
            for chunk_name in chunk_object_names:
                response = self._client.get_object(self._bucket, chunk_name)
                try:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        total += len(block)
                        hasher.update(block)
                        merged.write(block)
                finally:
                    response.close()
                    response.release_conn()

            if total != expected_size:
                raise ValueError(
                    f"Merged size mismatch: expected {expected_size}, got {total}."
                )
            if expected_md5 and hasher.hexdigest().lower() != expected_md5.lower():
                raise ValueError("Merged file MD5 mismatch.")

            merged.seek(0)
            self.upload_fileobj(
                object_name=object_name,
                data=merged,
                length=total,
                content_type="application/pdf",
            )
        return object_name

    def remove_objects(self, object_names: Sequence[str]) -> None:
        """Best-effort removal for temporary chunk objects."""
        for object_name in object_names:
            try:
                self._client.remove_object(self._bucket, object_name)
            except S3Error as exc:
                logger.warning("Failed to remove MinIO object {}: {}", object_name, exc)

    # ---------- presigned URL ----------

    def presigned_get_url(
        self,
        object_name: str,
        expires: timedelta = timedelta(hours=1),
    ) -> str:
        return self._client.presigned_get_object(
            bucket_name=self._bucket,
            object_name=object_name,
            expires=expires,
        )

    # ---------- properties ----------

    @property
    def bucket(self) -> str:
        return self._bucket


_minio_singleton: Optional[MinIOService] = None


def get_minio_service() -> MinIOService:
    """Process-wide singleton accessor (used both by API and worker)."""
    global _minio_singleton
    if _minio_singleton is None:
        _minio_singleton = MinIOService()
    return _minio_singleton
