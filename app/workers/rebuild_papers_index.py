"""Rebuild paper-level profiles for existing parsed tasks.

Run with:
    python -m app.workers.rebuild_papers_index
"""

from __future__ import annotations

import asyncio

from app.core.logging import logger, setup_logging
from app.core.schemas import ParsedDocument, TaskStatus
from app.services.storage.minio import get_minio_service
from app.services.papers.index import upsert_paper_profile
from app.services.papers.profile import extract_paper_profile
from app.services.tasks import list_tasks
from app.services.storage.es import ensure_indices


async def _rebuild_one(task) -> None:
    """重新抽取单篇论文画像，并输出方便人工检查的核心字段。"""
    if task.status != TaskStatus.SUCCEEDED or not task.parsed_object_name:
        logger.info("Skip task {}: status={} parsed={}", task.task_id, task.status, task.parsed_object_name)
        return

    try:
        payload = await asyncio.to_thread(get_minio_service().get_object_bytes, task.parsed_object_name)
        markdown = payload.decode("utf-8", errors="ignore")
        parsed = ParsedDocument(
            task_id=task.task_id,
            markdown=markdown,
            original_filename=task.original_filename,
            title=task.original_filename,
            num_pages=task.num_pages,
            num_tables=task.num_tables or 0,
            num_chars=len(markdown),
        )
        profile = await extract_paper_profile(parsed)
        await asyncio.to_thread(upsert_paper_profile, profile)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Rebuild paper profile failed for task {}: {}", task.task_id, exc)
        return

    print(
        "\t".join(
            [
                profile.title or profile.source_file or profile.paper_id,
                f"is_image_related={profile.is_image_related}",
                f"image_confidence={profile.image_confidence:.2f}",
                f"is_frequency_related={profile.is_frequency_related}",
                f"frequency_confidence={profile.frequency_confidence:.2f}",
                f"method_tags={', '.join(profile.method_tags[:8])}",
            ]
        )
    )


async def main() -> None:
    """遍历历史任务并重写 papers_index。"""
    setup_logging()
    await asyncio.to_thread(ensure_indices)
    tasks = list_tasks(limit=10000)
    logger.info("Rebuilding papers index for {} task records", len(tasks))
    for task in tasks:
        await _rebuild_one(task)


if __name__ == "__main__":
    asyncio.run(main())
