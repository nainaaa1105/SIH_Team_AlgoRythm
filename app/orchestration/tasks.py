"""Per-cluster Celery tasks.

M1 owns the queue and the enrichment tasks that populate `fingerprints`
(facility attribution, land cover, population). The classifier/imagery/
forecast tasks (M2/M3/M4/M5) are declared here as thin dispatch stubs so
the fan-out wiring exists end-to-end even before their real
implementations land — swap the body of each stub for the real import
once that member's module exists, without changing how M1 enqueues them.
"""
import logging
from pathlib import Path

from sqlalchemy import text

from app.config import get_settings
from app.enrichment.landcover import sample_landcover_at_point
from app.enrichment.osm_facilities import NEAREST_FACILITY_SQL
from app.enrichment.population import sample_population_at_point
from app.orchestration.queue import celery_app

logger = logging.getLogger(__name__)

# Local paths for the one-time-loaded rasters (see scripts/bulk_load_landcover.py).
# Population raster path is analogous once WorldPop is downloaded — not
# scripted yet, see CHANGES_MEMBER1.md known-limitations.
_LANDCOVER_RASTER = "data/landcover/mcd12q1_india.tif"
_POPULATION_RASTER = "data/population/worldpop_india.tif"


@celery_app.task(name="tasks.enrich_cluster")
def enrich_cluster(cluster_id: int, lon: float, lat: float) -> dict:
    """M1's own enrichment step: facility attribution + land cover +
    population density, written into `fingerprints`. Runs first in the
    per-cluster fan-out since M2's feature matrix depends on it.
    """
    from app.db.models import Fingerprint
    from app.db.session import session_scope

    settings = get_settings()
    landcover = None
    if Path(_LANDCOVER_RASTER).exists():
        landcover = sample_landcover_at_point(_LANDCOVER_RASTER, lon, lat)

    with session_scope() as session:
        from app.db.models import Cluster
        cluster = session.get(Cluster, cluster_id)
        year = cluster.first_seen.year if cluster and cluster.first_seen else 2020
        # clamp year between 2018 and 2020
        year = max(2018, min(2020, year))
        
        pop_raster_path = f"data/population/worldpop_india_{year}.tif"
        population = None
        if Path(pop_raster_path).exists():
            population = sample_population_at_point(pop_raster_path, lon, lat)
            
        nearest = session.execute(
            text(NEAREST_FACILITY_SQL), {"cluster_id": cluster_id, "max_distance_m": 5000.0}
        ).first()

        fingerprint = session.get(Fingerprint, cluster_id) or Fingerprint(cluster_id=cluster_id)
        if landcover:
            fingerprint.pct_cropland = landcover["pct_cropland"]
            fingerprint.pct_forest = landcover["pct_forest"]
            fingerprint.pct_urban = landcover["pct_urban"]
        if population is not None:
            fingerprint.population_density = population
        if nearest is not None:
            fingerprint.nearest_facility_id = nearest.id
            fingerprint.facility_distance_m = nearest.distance_m
        session.merge(fingerprint)

    logger.info(
        "Enriched cluster %s (landcover=%s, nearest_facility_distance_m=%s)",
        cluster_id, bool(landcover), getattr(nearest, "distance_m", None) if nearest else None,
    )
    return {"cluster_id": cluster_id, "status": "enriched"}


@celery_app.task(name="tasks.dispatch_downstream_jobs")
def dispatch_downstream_jobs(cluster_id: int, lon: float, lat: float, optical_available: bool) -> None:
    """Fan out to the other members' per-cluster jobs once M1's own
    enrichment is done. Each `.delay(...)` call is a stub queue entry —
    the actual task bodies live in each member's own module and get
    registered under these names when merged in.
    """
    celery_app.send_task("tasks.m2_build_feature_vector", args=[cluster_id])
    celery_app.send_task("tasks.m3_spatial_attribution", args=[cluster_id, lon, lat])
    # M4 is dispatched unconditionally and decides internally what it can
    # do. Its work splits across two data sources with different
    # dependencies: the Sentinel-2 half genuinely needs a clear sky, but
    # the Dozier sub-pixel temperature retrieval uses the VIIRS I4/I5
    # brightness temperatures that come straight from FIRMS and needs no
    # optical imagery at all. Gating the whole task on the cloud flag
    # threw that away for precisely the clusters with the least other
    # evidence — most of India through the monsoon.
    celery_app.send_task(
        "tasks.m4_imagery_features", args=[cluster_id, lon, lat, optical_available]
    )
    celery_app.send_task("tasks.m5_rhythm_and_kalman", args=[cluster_id])
