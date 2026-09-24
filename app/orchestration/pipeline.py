"""The full ingestion cycle: fetch every configured satellite source ->
dedup -> cluster (reconciled against existing DB clusters) -> persist ->
cloud-fraction gate -> fan out per-cluster jobs.

This is what `scheduler.py`'s APScheduler job calls every
POLL_INTERVAL_MINUTES, and what `scripts/backfill_90day.py` calls in a
loop over historical date windows.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import Cluster, Hotspot
from app.db.session import session_scope
from app.ingestion import firms, geostationary_supplementary, himawari, sentinel3_frp
from app.ingestion.cloud_gate import confidence_decay, get_cloud_fraction, is_optically_available
from app.ingestion.cluster import ClusterAssignment, cluster_batch
from app.ingestion.dedup import exact_key_for_db_row, full_dedup
from app.orchestration.tasks import dispatch_downstream_jobs, enrich_cluster

logger = logging.getLogger(__name__)


def fetch_all_sources(settings: Optional[Settings] = None) -> List[Dict]:
    """Poll every configured feed. Each connector already isolates its
    own failures (returns [] rather than raising) so one source being
    down never blocks the others.
    """
    settings = settings or get_settings()
    records: List[Dict] = []
    records.extend(firms.fetch_all_sources(settings))
    records.extend(geostationary_supplementary.fetch_all(settings))
    records.extend(sentinel3_frp.fetch_all(settings))
    records.extend(himawari.fetch_all(settings))
    logger.info("Combined fetch across all sources: %d raw records", len(records))
    return records


def load_existing_hotspot_keys(session: Session, since: datetime) -> set:
    """Build the same-shaped key as `dedup.exact_key_for_db_row` for every
    hotspot already persisted since `since`, so an in-memory dedup pass
    can catch re-poll duplicates before they ever reach the DB (the
    table's own UNIQUE constraint + ON CONFLICT DO NOTHING is still the
    authoritative backstop, this is purely to avoid needless re-work
    upstream — e.g. re-running the cloud-fraction GEE lookup for a
    cluster that hasn't actually changed).
    """
    from geoalchemy2.shape import to_shape

    rows = session.execute(
        select(Hotspot.source, Hotspot.geom, Hotspot.acq_datetime).where(Hotspot.acq_datetime >= since)
    ).all()
    keys = set()
    for r in rows:
        point = to_shape(r.geom)
        keys.add(exact_key_for_db_row(r.source, point.x, point.y, r.acq_datetime))
    return keys


def load_active_cluster_centroids(session: Session) -> List[Tuple[int, float, float]]:
    """Return (cluster_id, lon, lat) for every active cluster, decoding
    each PostGIS WKBElement centroid via GeoAlchemy2's shapely helper.
    """
    from geoalchemy2.shape import to_shape

    clusters = session.execute(select(Cluster).where(Cluster.status == "active")).scalars().all()
    result = []
    for c in clusters:
        point = to_shape(c.centroid)
        result.append((c.id, point.x, point.y))
    return result


# CDSE answers one catalogue query in roughly a second, and a live
# cycle over India produces thousands of clusters. Serially that is the
# single slowest step in the whole pipeline, and it is pure network wait,
# so it parallelises cleanly. Kept modest because CDSE is a shared public
# service and `get_cloud_fraction` is disk-cached, so the steady-state
# run issues very few real requests anyway.
CLOUD_PREFETCH_WORKERS = 8


def _prefetch_cloud_fractions(
    assignments: List[ClusterAssignment],
    workers: int = CLOUD_PREFETCH_WORKERS,
) -> Dict[int, Optional[float]]:
    """Pre-compute cloud fractions for every cluster OUTSIDE any DB
    transaction. Each call hits CDSE/GEE over the network and can take
    seconds; doing that with a Postgres session open would pin a
    connection and its locks for the duration.

    Results are keyed by the assignment's index, so the mapping is
    identical to the serial version regardless of completion order.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not assignments:
        return {}

    def _one(item):
        index, assignment = item
        try:
            return index, get_cloud_fraction(
                assignment.centroid_lon, assignment.centroid_lat, assignment.last_seen
            )
        except Exception:  # noqa: BLE001
            # Unknown, not clear — `is_optically_available(None)` is False
            # and the evidence-weighting engine records the gap honestly.
            logger.warning(
                "Cloud-fraction lookup failed for cluster at (%s, %s)",
                assignment.centroid_lon, assignment.centroid_lat, exc_info=True,
            )
            return index, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_one, enumerate(assignments)))


def persist_records_and_clusters(
    session: Session, records: List[Dict], assignments: List[ClusterAssignment],
    cloud_fractions: Dict[int, Optional[float]],
) -> List[Tuple[int, float, float, bool]]:
    """Write hotspots + clusters, return (cluster_id, lon, lat,
    optical_available) tuples for every touched cluster, for the caller
    to fan out downstream jobs against.

    Cloud fractions must be pre-computed by the caller (via
    ``_prefetch_cloud_fractions``) so no network I/O happens while
    the DB session is open.
    """
    from geoalchemy2.shape import from_shape, to_shape
    from shapely.geometry import Point

    touched_clusters: List[Tuple[int, float, float, bool]] = []

    for idx_a, assignment in enumerate(assignments):
        if assignment.cluster_id is None:
            cluster = Cluster(
                centroid=from_shape(Point(assignment.centroid_lon, assignment.centroid_lat), srid=4326),
                first_seen=assignment.first_seen,
                last_seen=assignment.last_seen,
                n_detections=len(assignment.record_indices),
            )
            session.add(cluster)
            session.flush()  # populate cluster.id
            cluster_id = cluster.id
        else:
            cluster = session.get(Cluster, assignment.cluster_id)
            # Recompute the centroid as the running weighted average of the
            # cluster's old detections and this batch's new ones, instead
            # of leaving it frozen at whichever detection happened to
            # create the cluster. A stale centroid is the anchor
            # match_to_existing_clusters compares tomorrow's fresh
            # detection against — day-to-day satellite geolocation jitter
            # (different sensor, different scan angle) can drift a real,
            # recurring fire's reported pixel centre by a couple hundred
            # metres each day, and once that accumulated drift from a
            # never-updated anchor exceeds dbscan_eps_degrees (~500m), the
            # next day's detection stops matching and starts a brand new
            # cluster next to the old one — the exact "same factory,
            # multiple nearby dots across days" bug this fixes.
            old_point = to_shape(cluster.centroid)
            old_n = cluster.n_detections or 0
            new_n = len(assignment.record_indices)
            total_n = old_n + new_n
            if total_n > 0:
                merged_lon = (old_point.x * old_n + assignment.centroid_lon * new_n) / total_n
                merged_lat = (old_point.y * old_n + assignment.centroid_lat * new_n) / total_n
                cluster.centroid = from_shape(Point(merged_lon, merged_lat), srid=4326)
            cluster.last_seen = max(cluster.last_seen, assignment.last_seen) if cluster.last_seen else assignment.last_seen
            cluster.n_detections = total_n
            cluster_id = cluster.id

        cloud_fraction = cloud_fractions.get(idx_a)
        cluster.cloud_fraction = cloud_fraction
        cluster.optical_available = is_optically_available(cloud_fraction)

        for idx in assignment.record_indices:
            record = records[idx]
            base_confidence = record.get("confidence")
            if base_confidence is None:
                base_confidence = 0.5  # unknown-confidence fallback; a legitimate 0.0 must NOT hit this branch
            effective_confidence = confidence_decay(base_confidence, cloud_fraction)
            stmt = (
                pg_insert(Hotspot)
                .values(
                    source=record["source"],
                    geom=from_shape(Point(record["lon"], record["lat"]), srid=4326),
                    acq_datetime=record["acq_datetime"],
                    frp=record.get("frp"),
                    brightness=record.get("brightness"),
                    confidence=effective_confidence,
                    daynight=record.get("daynight"),
                    scan=record.get("scan"),
                    track=record.get("track"),
                    raw_payload=record.get("raw_payload"),
                    cluster_id=cluster_id,
                )
                .on_conflict_do_nothing(constraint="uq_hotspot_identity")
            )
            session.execute(stmt)

        # The cluster's own (now merged, for an existing cluster) centroid,
        # not just this batch's — downstream enrichment (facility
        # attribution, land cover, WUI distance) should key off the same
        # authoritative position everything else about the cluster uses.
        dispatch_point = to_shape(cluster.centroid)
        touched_clusters.append(
            (cluster_id, dispatch_point.x, dispatch_point.y, cluster.optical_available)
        )

    return touched_clusters


def run_ingestion_cycle(settings: Optional[Settings] = None, dispatch_jobs: bool = True) -> Dict:
    """One full poll -> dedup -> cluster -> persist -> fan-out cycle."""
    from datetime import timezone

    settings = settings or get_settings()
    raw_records = fetch_all_sources(settings)

    with session_scope() as session:
        since = min((r["acq_datetime"] for r in raw_records), default=datetime.now(timezone.utc))
        existing_keys = load_existing_hotspot_keys(session, since)
        deduped = full_dedup(raw_records, existing_keys=existing_keys)

        existing_clusters = load_active_cluster_centroids(session)
        assignments = cluster_batch(deduped, existing_clusters, settings)

    # --- network I/O happens here, with NO transaction open ---
    cloud_fractions = _prefetch_cloud_fractions(assignments)

    # --- second short transaction for the actual writes ---
    with session_scope() as session:
        touched = persist_records_and_clusters(session, deduped, assignments, cloud_fractions)

    if dispatch_jobs:
        for cluster_id, lon, lat, optical_available in touched:
            enrich_cluster.delay(cluster_id, lon, lat)
            dispatch_downstream_jobs.delay(cluster_id, lon, lat, optical_available)

    # Push live updates to any open dashboards via WebSocket.
    _notify_dashboards(touched)

    logger.info("Ingestion cycle complete: %d raw -> %d deduped -> %d clusters touched",
                len(raw_records), len(deduped), len(touched))
    return {"raw": len(raw_records), "deduped": len(deduped), "clusters_touched": len(touched)}


def broker_is_reachable(timeout_seconds: float = 1.0) -> bool:
    """Can we actually reach the Celery broker right now?

    `run_ingestion_cycle(dispatch_jobs=True)` ends in `.delay()` calls.
    With no broker those do not fail fast — Celery retries the connection
    and eventually raises "Retry limit exceeded", by which point the
    ingestion writes have already committed and the caller gets a bare
    500 for work that partly succeeded. Checking first lets the caller
    pick a path that will actually complete.
    """
    try:
        import redis

        client = redis.Redis.from_url(
            get_settings().redis_url,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
        )
        client.ping()
        return True
    except Exception:  # noqa: BLE001
        return False


def celery_worker_available(timeout_seconds: float = 1.5) -> bool:
    """A reachable broker is not the same as a worker that will ever
    drain it. A deployment can have Redis provisioned with no worker
    service running against it at all (Dockerfile.worker exists but was
    never deployed as its own service) -- in that case `.delay()` calls
    succeed immediately and then sit in the queue forever, un-consumed,
    with no error raised anywhere. Pinging for a live worker (not just a
    live broker) is what actually answers "will this get processed".
    """
    if not broker_is_reachable(timeout_seconds):
        return False
    try:
        from app.orchestration.queue import celery_app

        pong = celery_app.control.ping(timeout=timeout_seconds)
        return bool(pong)
    except Exception:  # noqa: BLE001
        return False


def run_cycle_safely(classify: bool = True) -> Dict:
    """Pick the dispatch path that will actually get consumed, not just
    accepted -- the single decision both /ingest/run and the scheduler's
    periodic job share, so there is exactly one place this is decided.

    With a broker AND a live worker to drain it, this fans per-cluster
    work out over Celery, matching the intended deployment shape. Without
    a live worker it runs that same work in-process via `local_pipeline`
    instead -- see `celery_worker_available`'s docstring for why a
    reachable-but-unconsumed broker would otherwise silently classify
    nothing. `classify=False` ingests only, leaving classification for
    later.
    """
    if celery_worker_available():
        summary = run_ingestion_cycle()
        summary["mode"] = "celery"
        return summary

    logger.info("No live Celery worker — running the cycle in-process")

    if not classify:
        summary = run_ingestion_cycle(None, False)
        summary["mode"] = "in-process (ingest only)"
        return summary

    from app.orchestration.local_pipeline import run_cycle

    summary = run_cycle()
    summary["mode"] = "in-process"
    return summary


def _notify_dashboards(touched: List[Tuple[int, float, float, bool]]) -> None:
    """Best-effort WebSocket push to open dashboards.

    Runs in a sync context (Celery/scheduler), so uses asyncio.run().
    Failures are swallowed — WebSocket is optional, ingestion is not.
    """
    if not touched:
        return
    try:
        import asyncio

        from gateway.live import manager

        async def _broadcast():
            await manager.broadcast({
                "type": "ingestion_complete",
                "clusters_touched": len(touched),
                "cluster_ids": [cid for cid, _, _, _ in touched],
            })

        # If an event loop is already running (inside uvicorn), schedule
        # the coroutine on it. Otherwise (Celery worker), use asyncio.run.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_broadcast())
        except RuntimeError:
            asyncio.run(_broadcast())
    except Exception:
        logger.debug("WebSocket notification failed (non-critical)", exc_info=True)
