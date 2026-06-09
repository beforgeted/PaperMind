"""Standalone Kafka consumer worker.

Run with:
    python -m app.workers.consumer

Lifecycle per message:
    pending -> parsing -> (Phase 4 will append: indexing -> succeeded)
                       \\-> failed (with error message)

For Phase 3 we stop at "parsed": Markdown is uploaded back to MinIO and the
task record is marked SUCCEEDED. Phase 4 will replace the success branch with
chunk + embed + index, transitioning through INDEXING first.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Windows DLL 顺序兜底（必须先于任何重型 import）
#
# numpy / scipy / sklearn / langchain_elasticsearch 等会各自携带一份
# Intel OpenMP (`libiomp5md.dll`)。如果它们先于 PyTorch 加载，再 `import torch`
# 就会因 OMP 运行时冲突触发 `[WinError 1114] c10.dll 初始化例程失败`。
# 两层防护，缺一不可：
#   1) 提前 `import torch`，让它的 DLL 抢占加载顺序；
#   2) 设置 `KMP_DUPLICATE_LIB_OK=TRUE`，给 Intel OpenMP 的重复加载开后门。
# 这只对 worker 进程生效，API 进程无需付出该代价。
# ---------------------------------------------------------------------------
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402,F401  必须最先导入

import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from aiokafka import AIOKafkaConsumer  # noqa: E402
from aiokafka.errors import KafkaError  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.schemas import ParseTask, TaskStatus  # noqa: E402
from app.services.docling import DoclingParseError, get_docling_service  # noqa: E402
from app.services.indexing import run_indexing  # noqa: E402
from app.services.papers.index import upsert_paper_profile  # noqa: E402
from app.services.papers.profile import extract_paper_profile  # noqa: E402
from app.services.minio_service import get_minio_service  # noqa: E402
from app.services.task_svc import update_task  # noqa: E402

logger = logging.getLogger(__name__)

_shutdown = asyncio.Event()


def _install_signal_handlers() -> None:
    def _handle(signum, _frame):
        logger.info("Received signal {}, shutting down worker…", signum)
        _shutdown.set()

    # Windows: only SIGINT/SIGTERM are supported here.
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):
            pass


async def _process_one(task: ParseTask) -> None:
    """Pull PDF -> Docling parse -> store Markdown back to MinIO."""
    logger.info("Processing task {} (object={})", task.task_id, task.object_name)
    update_task(task.task_id, status=TaskStatus.PARSING, message="Downloading PDF…")

    minio = get_minio_service()
    docling = get_docling_service()

    with tempfile.TemporaryDirectory(prefix=f"papermind-{task.task_id}-") as tmpdir:
        local_pdf = Path(tmpdir) / task.original_filename
        # 1) Pull PDF from MinIO
        await asyncio.to_thread(minio.download_to_path, task.object_name, local_pdf)

        # 2) Docling parse (CPU-bound -> threadpool)
        update_task(task.task_id, message="Running Docling…")
        try:
            parsed = await asyncio.to_thread(
                docling.parse_pdf,
                local_pdf,
                task.task_id,
                task.original_filename,
            )
        except DoclingParseError as exc:
            update_task(
                task.task_id,
                status=TaskStatus.FAILED,
                error=f"docling: {exc}",
                message="Parsing failed.",
            )
            return

        # 3) Persist Markdown back to MinIO so Phase 4 can pick it up.
        parsed_object = f"parsed/{task.task_id}/document.md"
        await asyncio.to_thread(
            minio.upload_bytes,
            parsed_object,
            parsed.markdown.encode("utf-8"),
            "text/markdown; charset=utf-8",
        )

    # 4) Chunk + embed + index in Elasticsearch.
    update_task(
        task.task_id,
        status=TaskStatus.INDEXING,
        message="Chunking, embedding and indexing…",
        parsed_object_name=parsed_object,
        num_pages=parsed.num_pages,
        num_tables=parsed.num_tables,
    )
    try:
        stats = await asyncio.to_thread(run_indexing, parsed)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Indexing failed for task {}: {}", task.task_id, exc)
        update_task(
            task.task_id,
            status=TaskStatus.FAILED,
            error=f"indexing: {exc}",
            message="Indexing failed.",
        )
        return

    # 5) 论文级画像写入（失败不影响主流程）。
    try:
        profile = await extract_paper_profile(parsed)
        await asyncio.to_thread(upsert_paper_profile, profile)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Paper profile upsert skipped for task {}: {}", task.task_id, exc)

    update_task(
        task.task_id,
        status=TaskStatus.SUCCEEDED,
        message="Parsed, embedded and indexed.",
        num_parents=stats.num_parents,
        num_children=stats.num_children,
    )
    logger.info(
        "Task {} done — {} parents / {} children indexed",
        task.task_id,
        stats.num_parents,
        stats.num_children,
    )


async def _consume_loop() -> None:
    consumer = AIOKafkaConsumer(
        settings.kafka_topic_parse,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_group_id,
        client_id=f"{settings.kafka_client_id}-worker",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    await consumer.start()
    logger.info(
        "Kafka consumer started: topic={} group={} bootstrap={}",
        settings.kafka_topic_parse,
        settings.kafka_group_id,
        settings.kafka_bootstrap_servers,
    )

    try:
        while not _shutdown.is_set():
            try:
                # short poll so we can react to shutdown
                batch = await consumer.getmany(timeout_ms=1000, max_records=1)
            except KafkaError as exc:
                logger.exception("Kafka poll failed: {}", exc)
                await asyncio.sleep(2)
                continue

            for _tp, messages in batch.items():
                for msg in messages:
                    try:
                        task = ParseTask.model_validate(msg.value)
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "Skipping unparseable message at offset {}: {}",
                            msg.offset,
                            exc,
                        )
                        await consumer.commit()
                        continue

                    try:
                        await _process_one(task)
                    except Exception as exc:  # noqa: BLE001
                        # Mark FAILED and commit anyway to avoid poison-message loops.
                        logger.exception("Worker error on task {}: {}", task.task_id, exc)
                        update_task(
                            task.task_id,
                            status=TaskStatus.FAILED,
                            error=str(exc),
                            message="Unhandled worker error.",
                        )

                    await consumer.commit()
    finally:
        await consumer.stop()
        logger.info("Kafka consumer stopped.")


def _warmup_main_thread() -> None:
    """主线程预热 Docling，使首条任务的延迟更稳定。

    `torch` 已在模块顶部完成首次 import（见文件开头的 Windows DLL 兜底注释），
    这里只需把 Docling 的 `DocumentConverter` 提前实例化好。
    """
    logger.info("Warming up Docling DocumentConverter on main thread…")
    logger.debug("torch version={} cuda={}", torch.__version__, torch.version.cuda)
    get_docling_service().warmup()
    logger.info("Warmup done (Docling ready).")


async def main() -> None:
    from app.core.logging import setup_logging

    setup_logging()
    _install_signal_handlers()
    logger.info("PaperMind worker starting (env={})", settings.app_env)
    try:
        _warmup_main_thread()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Warmup failed: {}", exc)
        raise
    try:
        await _consume_loop()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal worker error: {}", exc)
        raise


if __name__ == "__main__":
    asyncio.run(main())
