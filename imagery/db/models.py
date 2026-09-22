"""M4's tables, declared against M1's shared `Base`.

Same discipline as M2 and M3: one database, one MetaData, one Alembic
history, so foreign keys resolve and autogenerate sees the whole schema.

Note what is deliberately *not* here: the fused class probabilities. M2's
`classifications.image_probabilities` column already exists for that, so
M4 hands its verdict to `tasks.m2_classify` rather than keeping a second
copy of the fused result. `image_predictions` stores only what the CNN
said on its own, which is what the event card needs to show image
evidence independently of the fusion.
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
    Text,
)

from app.db.models import Base  # M1's declarative base — shared MetaData


class Sentinel2Patch(Base):
    """One fetched optical patch for a cluster.

    Multiple rows per cluster are expected: Sentinel-2 revisits every ~5
    days, and a persistent source accumulates a time series of patches
    that makes before/after comparison possible.
    """

    __tablename__ = "sentinel2_patches"

    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)
    file_path = Column(Text, nullable=True)        # .npz of the band stack
    thumbnail_path = Column(Text, nullable=True)   # RGB preview for M6's event card
    acquired_at = Column(DateTime(timezone=True), nullable=True)
    bbox = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=True)
    cloud_percentage = Column(Float, nullable=True)
    bands = Column(JSON, nullable=True)            # band order in the stored array
    width_px = Column(Integer, nullable=True)
    height_px = Column(Integer, nullable=True)
    source = Column(String(40), nullable=True)     # e.g. GEE_S2_SR_HARMONIZED
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class ImagePrediction(Base):
    """What the image model alone concluded, before fusion."""

    __tablename__ = "image_predictions"

    cluster_id = Column(Integer, ForeignKey("clusters.id"), primary_key=True)
    patch_id = Column(Integer, ForeignKey("sentinel2_patches.id"), nullable=True)
    class_probabilities = Column(JSON, nullable=True)
    predicted_class = Column(String(30), nullable=True)
    confidence = Column(Float, nullable=True)
    model_version = Column(String(20), nullable=True)
    # Observed plume direction, for cross-checking M3's modelled bearing.
    smoke_detected = Column(Boolean, default=False)
    smoke_bearing_deg = Column(Float, nullable=True)
    smoke_coverage = Column(Float, nullable=True)
    plume_agreement = Column(JSON, nullable=True)   # verdict vs M3's plume
    computed_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class ThermalRetrieval(Base):
    """Dozier sub-pixel retrieval, with the assumptions it depended on.

    The background temperature and its provenance are stored alongside
    the answer because the retrieval is only interpretable against the
    background it assumed — and that background is estimated, not
    measured (FIRMS ships no non-fire neighbours).
    """

    __tablename__ = "thermal_retrievals"

    cluster_id = Column(Integer, ForeignKey("clusters.id"), primary_key=True)
    fire_temperature_k = Column(Float, nullable=True)
    fire_fraction = Column(Float, nullable=True)
    fire_area_m2 = Column(Float, nullable=True)
    background_temperature_k = Column(Float, nullable=True)
    background_source = Column(String(30), nullable=True)
    sensor = Column(String(20), nullable=True)
    converged = Column(Boolean, default=False)
    reason = Column(Text, nullable=True)
    n_detections_used = Column(Integer, nullable=True)
    # Bracket from re-solving at the edges of the background's own
    # uncertainty. The inversion is badly ill-conditioned for small hot
    # fires, so a bare temperature would imply a precision this method
    # does not have — `well_constrained` says whether to trust the number.
    temperature_low_k = Column(Float, nullable=True)
    temperature_high_k = Column(Float, nullable=True)
    well_constrained = Column(Boolean, nullable=True)
    computed_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
