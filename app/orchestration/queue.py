"""Celery app definition — the task queue/orchestrator that fans work out
per cluster_id to M2 (features/classifier), M3 (spatial joins/plume),
M4 (Dozier/imagery), M5 (Kalman/rhythm) once M1's ingestion cycle creates
or updates a cluster.
"""
from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "sih_ingestion",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_default_queue="cluster_events",
)
