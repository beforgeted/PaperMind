"""Redis Bitmap-backed state for multipart uploads."""

from __future__ import annotations

from typing import Optional

from redis import Redis

from app.core.config import settings


class UploadStateError(RuntimeError):
    """Raised when multipart upload state is missing or inconsistent."""


class RedisUploadState:
    """Track uploaded chunk indexes with Redis Bitmap and metadata hashes."""

    def __init__(self) -> None:
        self._client = Redis.from_url(settings.redis_url, decode_responses=True)
        self._ttl = settings.upload_state_ttl_seconds

    def _meta_key(self, upload_id: str) -> str:
        return f"upload:{upload_id}:meta"

    def _bitmap_key(self, upload_id: str) -> str:
        return f"upload:{upload_id}:bitmap"

    def _chunks_key(self, upload_id: str) -> str:
        return f"upload:{upload_id}:chunks"

    def init_upload(
        self,
        *,
        upload_id: str,
        task_id: str,
        filename: str,
        object_name: str,
        total_size: int,
        total_chunks: int,
        file_md5: Optional[str],
    ) -> None:
        meta_key = self._meta_key(upload_id)
        self._client.hset(
            meta_key,
            mapping={
                "task_id": task_id,
                "filename": filename,
                "object_name": object_name,
                "total_size": str(total_size),
                "total_chunks": str(total_chunks),
                "file_md5": file_md5 or "",
                "status": "uploading",
            },
        )
        self._refresh_ttl(upload_id)

    def get_meta(self, upload_id: str) -> dict:
        meta = self._client.hgetall(self._meta_key(upload_id))
        if not meta:
            raise UploadStateError(f"Upload {upload_id} not found or expired.")
        return meta

    def mark_chunk(
        self,
        *,
        upload_id: str,
        chunk_index: int,
        chunk_md5: str,
        chunk_size: int,
        object_name: str,
    ) -> int:
        self.get_meta(upload_id)
        self._client.hset(
            self._chunks_key(upload_id),
            str(chunk_index),
            f"{chunk_md5}:{chunk_size}:{object_name}",
        )
        self._client.setbit(self._bitmap_key(upload_id), chunk_index, 1)
        self._refresh_ttl(upload_id)
        return self.uploaded_count(upload_id)

    def uploaded_count(self, upload_id: str) -> int:
        return int(self._client.bitcount(self._bitmap_key(upload_id)))

    def uploaded_chunks(self, upload_id: str) -> list[int]:
        total_chunks = int(self.get_meta(upload_id)["total_chunks"])
        bitmap_key = self._bitmap_key(upload_id)
        return [
            index
            for index in range(total_chunks)
            if int(self._client.getbit(bitmap_key, index)) == 1
        ]

    def chunk_object_names(self, upload_id: str) -> list[str]:
        meta = self.get_meta(upload_id)
        total_chunks = int(meta["total_chunks"])
        raw = self._client.hgetall(self._chunks_key(upload_id))
        names: list[str] = []
        for index in range(total_chunks):
            value = raw.get(str(index))
            if not value:
                raise UploadStateError(f"Chunk {index} is missing.")
            names.append(value.split(":", 2)[2])
        return names

    def mark_completed(self, upload_id: str) -> None:
        self._client.hset(self._meta_key(upload_id), "status", "completed")
        self._refresh_ttl(upload_id)

    def cleanup(self, upload_id: str) -> None:
        self._client.delete(
            self._meta_key(upload_id),
            self._bitmap_key(upload_id),
            self._chunks_key(upload_id),
        )

    def _refresh_ttl(self, upload_id: str) -> None:
        for key in (
            self._meta_key(upload_id),
            self._bitmap_key(upload_id),
            self._chunks_key(upload_id),
        ):
            self._client.expire(key, self._ttl)


_upload_state_singleton: Optional[RedisUploadState] = None


def get_upload_state() -> RedisUploadState:
    global _upload_state_singleton
    if _upload_state_singleton is None:
        _upload_state_singleton = RedisUploadState()
    return _upload_state_singleton
