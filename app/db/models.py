"""ORM models for the shared schema: hotspots, clusters, facilities,
fingerprints, alerts.

This schema is the hand-off contract for the whole team (see
SIH26162_team_task_division.md) — M2-M5's feature engineering keys off
`clusters.id` / `fingerprints`, and M6's dashboard reads `hotspots` +
`clusters` + `facilities` directly. Do not rename columns here without
a team heads-up.
"""
from datetime import datetime, timezone

from geoalchemy2 import Geometry
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Hotspot(Base):
    """One row per raw satellite detection, before clustering.

    `source` keeps sensor provenance even after dedup/clustering, because
    the evidence-weighting engine (M2) needs to know which sensor(s)
    backed each cluster.
    """

    __tablename__ = "hotspots"
    __table_args__ = (
        UniqueConstraint("source", "geom", "acq_datetime", name="uq_hotspot_identity"),
        CheckConstraint("daynight in ('D','N') OR daynight IS NULL", name="ck_daynight"),
    )

    id = Column(Integer, primary_key=True)
    source = Column(String(20), nullable=False)  # VIIRS_SNPP, VIIRS_NOAA20, MODIS, INSAT3DS, SENTINEL3_FRP, HIMAWARI
    geom = Column(Geometry(geometry_type="POINT", srid=4326), nullable=False)
    acq_datetime = Column(DateTime(timezone=True), nullable=False)
    frp = Column(Float, nullable=True)          # Fire Radiative Power (MW)
    brightness = Column(Float, nullable=True)   # brightness temperature (K)
    confidence = Column(Float, nullable=True)   # normalized 0-1 (raw scales vary per sensor)
    daynight = Column(String(1), nullable=True)
    scan = Column(Float, nullable=True)
    track = Column(Float, nullable=True)
    raw_payload = Column(JSON, nullable=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=True)
    ingested_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    cluster = relationship("Cluster", back_populates="hotspots")


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True)
    centroid = Column(Geometry(geometry_type="POINT", srid=4326), nullable=False)
    first_seen = Column(DateTime(timezone=True), nullable=True)
    last_seen = Column(DateTime(timezone=True), nullable=True)
    n_detections = Column(Integer, default=0)
    cloud_fraction = Column(Float, nullable=True)
    optical_available = Column(Boolean, default=False)
    status = Column(String(20), default="active")  # active/stale/resolved
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # WUI (wildland-urban interface) proximity — written by
    # geospatial.tasks.wui_threat, read by the dashboard's detections/
    # alerts feeds and by GET /geospatial/wui/{id}. Deliberately scalar
    # columns on `clusters` rather than a joined table: the dashboard's
    # hot path already loads one Cluster row per item, and "is this one
    # thing true right now" is exactly what belongs next to `status`,
    # not behind an extra join. Unlike `cluster_features` (which has a
    # strict per-member column-ownership test because five packages
    # write concurrent columns on it), `clusters` has no such contract —
    # M1 already owns the whole row, so M3 writing four more columns
    # here does not cross an ownership boundary the tests protect.
    wui_threat = Column(Boolean, default=False)
    wui_distance_m = Column(Float, nullable=True)
    wui_eta_hours = Column(Float, nullable=True)
    wui_bearing_deg = Column(Float, nullable=True)
    wui_threatened_asset = Column(String(255), nullable=True)

    # Crown-fire FRP threshold detection — written by
    # geospatial.tasks.crown_fire (geospatial/crown_fire.py). Independent
    # of WUI: an FRP spike is diagnostic of fire behaviour on its own,
    # for any class, not just a spreading one near a settlement.
    is_crown_fire = Column(Boolean, default=False)

    # WFDSS-inspired Resource Demand Index — written by
    # geospatial.tasks.resource_demand (geospatial/decision_engine.py).
    # Ranks this cluster against every other active cluster for
    # resource allocation; reuses the WUI/crown-fire columns above
    # rather than re-deriving proximity or thermal history.
    rdi_score = Column(Float, default=0.0)
    coa_type = Column(String(100), default="MONITOR_ECOLOGICAL_BENEFIT")

    # The real footprint-area figure resource_demand already computes
    # internally (geospatial.decision_engine.footprint_area_ha) but
    # never persisted — geospatial.suppression needs it for every
    # cluster at once (the dashboard's Fire Suppression tab lists all
    # of them), and recomputing it live for hundreds of clusters on
    # every page load is wasteful when resource_demand already derived
    # it once per cycle.
    area_ha = Column(Float, nullable=True)

    hotspots = relationship("Hotspot", back_populates="cluster")
    fingerprint = relationship("Fingerprint", back_populates="cluster", uselist=False)
    alerts = relationship("Alert", back_populates="cluster")


class Facility(Base):
    __tablename__ = "facilities"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=True)
    facility_type = Column(String(50), nullable=True)  # refinery/steel/power/mine/LNG/flare
    source = Column(String(20), nullable=True)  # OSM or GEM
    geom = Column(Geometry(geometry_type="GEOMETRY", srid=4326), nullable=False)  # point or polygon
    prior_weight = Column(Float, default=1.0)
    facility_metadata = Column("metadata", JSON, nullable=True)


class Fingerprint(Base):
    __tablename__ = "fingerprints"

    cluster_id = Column(Integer, ForeignKey("clusters.id"), primary_key=True)
    pct_cropland = Column(Float, nullable=True)
    pct_forest = Column(Float, nullable=True)
    pct_urban = Column(Float, nullable=True)
    nearest_facility_id = Column(Integer, ForeignKey("facilities.id"), nullable=True)
    facility_distance_m = Column(Float, nullable=True)
    population_density = Column(Float, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    cluster = relationship("Cluster", back_populates="fingerprint")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)
    severity = Column(String(20), nullable=True)
    message = Column(String, nullable=True)
    dispatched_at = Column(DateTime(timezone=True), nullable=True)
    channel = Column(String(20), nullable=True)  # sms/email/webhook
    # Added for the emergency SMS dispatch feature (test mode) —
    # app/notifications/sms_dispatch.py is the first real writer of this
    # table. `recipient` is the phone number actually used, not the
    # incident's — the recipient is test-mode-only for now, see
    # app/notifications/sms_recipient.py.
    recipient = Column(String(32), nullable=True)
    status = Column(String(20), nullable=True)  # sent/failed
    provider = Column(String(20), nullable=True)
    provider_message_id = Column(String(64), nullable=True)
    error = Column(String, nullable=True)

    cluster = relationship("Cluster", back_populates="alerts")


class User(Base):
    """A dashboard operator account. See app/auth/ for the hashing and
    session-token logic that reads/writes this table — nothing here
    stores a plaintext password.

    `role` gates which login screen the account works on ("admin" sees
    all-India data, "state" is scoped to `state` on the dashboard) — see
    app/auth/routes.py::login, which refuses a login whose requested
    role doesn't match the account's own. `state` is populated only for
    role="state" and is validated against the real boundary resolver's
    state list (geospatial.admin_boundaries.state_names), not a
    hand-typed one, so it can never hold a state the dashboard can't
    actually filter on.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(120), nullable=False)
    government_id = Column(String(64), nullable=False)
    role = Column(String(20), nullable=False)  # "admin" | "state"
    state = Column(String(64), nullable=True)  # only set when role == "state"
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
