"""M2's tables, declared against M1's shared `Base`.

Importing M1's Base (rather than declaring a second one) means these
tables join the same MetaData, so Alembic autogenerate sees them and
foreign keys to `clusters` resolve properly. The migration that creates
them (0002) lives in M1's migration chain for the same reason: there is
one database and one version history, not one per team member.

`cluster_features` is the shared 28-feature matrix the team task
division describes — M2 creates and owns the table, but M3/M4/M5 each
UPDATE their own columns on it from their own Celery tasks. Column
ownership is annotated per-field below.
"""
from datetime import datetime

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


class ClusterFeatures(Base):
    """One row per cluster: the assembled 28-feature matrix + extras."""

    __tablename__ = "cluster_features"

    cluster_id = Column(Integer, ForeignKey("clusters.id"), primary_key=True)

    # --- CONTEXT group: snapshotted from M1's `fingerprints` at assembly
    # time. Copied rather than joined so a training row is a stable record
    # of what was known when it was classified, even if M1 later refreshes
    # the land-cover raster or the facility registry.
    pct_cropland = Column(Float, nullable=True)
    pct_forest = Column(Float, nullable=True)
    pct_urban = Column(Float, nullable=True)
    facility_distance_m = Column(Float, nullable=True)
    facility_prior_weight = Column(Float, nullable=True)
    facility_type = Column(String(50), nullable=True)
    population_density = Column(Float, nullable=True)
    near_facility = Column(Float, nullable=True)

    # --- THERMAL group (M2)
    frp_mean = Column(Float, nullable=True)
    frp_max = Column(Float, nullable=True)
    frp_std = Column(Float, nullable=True)
    frp_zscore = Column(Float, nullable=True)
    brightness_mean = Column(Float, nullable=True)
    brightness_max = Column(Float, nullable=True)
    confidence_mean = Column(Float, nullable=True)

    # --- TEMPORAL group (M2)
    persistence_days = Column(Float, nullable=True)
    n_detections = Column(Float, nullable=True)
    detections_per_day = Column(Float, nullable=True)
    night_fraction = Column(Float, nullable=True)
    source_diversity = Column(Float, nullable=True)

    # --- SPATIAL group (M2)
    spatial_extent_km = Column(Float, nullable=True)
    spatial_growth_rate = Column(Float, nullable=True)

    # --- IMAGERY group (M4 writes these)
    dozier_temp = Column(Float, nullable=True)
    ndvi = Column(Float, nullable=True)
    ndbi = Column(Float, nullable=True)
    smoke_red_blue_ratio = Column(Float, nullable=True)

    # --- RHYTHM group (M5 writes these)
    shift_sharpness = Column(Float, nullable=True)
    weekend_suppression = Column(Float, nullable=True)
    kalman_time_to_critical = Column(Float, nullable=True)

    # --- M3 writes this. Stored for the dashboard/alerting only — NOT a
    # model feature, because a threat corridor is computed downstream of
    # classification and feeding it back would be target leakage.
    threat_corridor_present = Column(Boolean, nullable=True)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class Classification(Base):
    """The model's verdict for a cluster, with its explanation."""

    __tablename__ = "classifications"
    __table_args__ = (UniqueConstraint("cluster_id", name="uq_classification_cluster"),)

    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)
    predicted_class = Column(String(30), nullable=True)
    class_probabilities = Column(JSON, nullable=True)
    xgb_probabilities = Column(JSON, nullable=True)
    image_probabilities = Column(JSON, nullable=True)  # NULL when cloud-blocked
    fusion_weight_xgb = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)    # 0-100 headline for the UI
    shap_values = Column(JSON, nullable=True)
    explanation_method = Column(String(40), nullable=True)
    reasons = Column(JSON, nullable=True)
    evidence_weight_notes = Column(JSON, nullable=True)
    model_version = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class TrainingLabel(Base):
    """Ground truth for training, kept separate from live inference output."""

    __tablename__ = "training_labels"

    cluster_id = Column(Integer, ForeignKey("clusters.id"), primary_key=True)
    label = Column(String(30), nullable=False)
    label_source = Column(String(30), nullable=True)   # GGFR/FSI/AGRI_RULE/...
    label_confidence = Column(Float, nullable=True)
    reason = Column(String, nullable=True)
    is_ambiguous = Column(Boolean, default=False)      # contradictory rules -> analyst review
    verified_by_analyst = Column(Boolean, default=False)
    labeled_at = Column(DateTime(timezone=True), default=datetime.utcnow)
