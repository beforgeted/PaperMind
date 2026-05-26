"""Async Kafka producer for publishing parse tasks.

Lifecycle is owned by the FastAPI lifespan:
    - `await get_kafka_producer().start()` on app startup
    - `await get_kafka_producer().stop()`  on app shutdown

Worker side (Phase 3) uses a separate consumer in app/workers/consumer.py.
"""
from __future__ import annotations

import json
from typing import Optional

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

from app.core.config import settings
from app.core.logging import logger
from app.core.schemas import ParseTask


def _json_serializer(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")


class KafkaProducerService:
    """Wrapper around AIOKafkaProducer with lazy start + safe shutdown."""

    def __init__(self) -> None:
        self._producer: Optional[AIOKafkaProducer] = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            client_id=settings.kafka_client_id,
            value_serializer=_json_serializer,
            enable_idempotence=True,
            acks="all",
        )
        await self._producer.start()
        self._started = True
        logger.info(
            "Kafka producer started (bootstrap={}, client_id={})",
            settings.kafka_bootstrap_servers,
            settings.kafka_client_id,
        )

    async def stop(self) -> None:
        if not self._started or self._producer is None:
            return
        try:
            await self._producer.stop()
        finally:
            self._started = False
            self._producer = None
            logger.info("Kafka producer stopped")

    async def send_parse_task(self, task: ParseTask) -> None:
        if not self._started or self._producer is None:
            raise RuntimeError("Kafka producer is not started")

        payload = task.model_dump(mode="json")
        try:
            # Key by task_id so all events for a task land on the same partition.
            await self._producer.send_and_wait(
                topic=settings.kafka_topic_parse,
                key=task.task_id.encode("utf-8"),
                value=payload,
            )
            logger.info(
                "Published parse task {} to topic {}",
                task.task_id,
                settings.kafka_topic_parse,
            )
        except KafkaError as exc:
            logger.exception("Kafka send failed for task {}: {}", task.task_id, exc)
            raise


_producer_singleton: Optional[KafkaProducerService] = None


def get_kafka_producer() -> KafkaProducerService:
    """Process-wide singleton accessor."""
    global _producer_singleton
    if _producer_singleton is None:
        _producer_singleton = KafkaProducerService()
    return _producer_singleton
