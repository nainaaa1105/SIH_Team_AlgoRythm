"""M3's tables, declared against M1's shared `Base`.

Same rationale as M2's models: one database, one MetaData, one Alembic
history. Importing M1's Base means foreign keys to `clusters` and
`facilities` resolve, and autogenerate sees the whole schema.

M3's outputs are geometries, which is why they get real tables rather
than scalar columns on `cluster_features`. The one scalar M3 owns there
is `threat_corridor_present`, which M2's tests assert nobody else
writes.
"""
from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)

from app.db.models import Base  # M1's declarative base — shared MetaData


class FacilityAttribution(Base):
    """One row per candidate facility per cluster, ranked.

    Multiple rows per cluster on purpose: responders need the runners-up,
    not just the winner.
    """

    __tablename__ = "facility_attributions"
    __table_args__ = (
        UniqueConstraint("cluster_id", "facility_id", name="uq_attribution_cluster_facility"),
    )

    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)
    facility_id = Column(Integer, ForeignKey("facilities.id"), nullable=False)
    distance_m = Column(Float, nullable=True)
    inside_polygon = Column(Boolean, default=False)
    attribution_probability = Column(Float, nullable=True)
    rank = Column(Integer, nullable=True)
    computed_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class Plume(Base):
    """Downwind dispersion footprint for a cluster."""

    __tablename__ = "plumes"

    cluster_id = Column(Integer, ForeignKey("clusters.id"), primary_key=True)
    geom = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=True)
    wind_speed_ms = Column(Float, nullable=True)
    wind_direction_deg = Column(Float, nullable=True)   # meteorological: FROM
    downwind_bearing_deg = Column(Float, nullable=True)  # travel direction: TO
    wind_source = Column(String(20), nullable=True)      # open-meteo | fallback
    stability_class = Column(String(1), nullable=True)   # Pasquill A-F
    plume_rise_m = Column(Float, nullable=True)
    length_m = Column(Float, nullable=True)
    area_km2 = Column(Float, nullable=True)
    chemical_profile = Column(JSON, nullable=True)
    population_exposed = Column(Integer, nullable=True)
    population_known = Column(Boolean, default=False)
    computed_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class ThreatCorridor(Base):
    """Projected advance wedge for a spreading fire."""

    __tablename__ = "threat_corridors"

    cluster_id = Column(Integer, ForeignKey("clusters.id"), primary_key=True)
    geom = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=True)
    bearing_deg = Column(Float, nullable=True)
    half_angle_deg = Column(Float, nullable=True)
    length_km = Column(Float, nullable=True)
    spread_rate_km_day = Column(Float, nullable=True)
    trajectory_confidence = Column(Float, nullable=True)
    projection_hours = Column(Float, nullable=True)
    threatened_facilities = Column(JSON, nullable=True)   # [{facility_id, name, distance_km, eta_hours}]
    min_time_to_impact_hours = Column(Float, nullable=True)
    computed_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class EvacuationRoute(Base):
    """A road path out of a hazard polygon."""

    __tablename__ = "evacuation_routes"

    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)
    geom = Column(Geometry(geometry_type="LINESTRING", srid=4326), nullable=True)
    origin_name = Column(String(255), nullable=True)
    destination_name = Column(String(255), nullable=True)
    length_km = Column(Float, nullable=True)
    computed_at = Column(DateTime(timezone=True), default=datetime.utcnow)
