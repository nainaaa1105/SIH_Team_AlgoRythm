"""M5's tables, declared against M1's shared `Base`.

Same discipline as M2, M3 and M4: one database, one MetaData, one Alembic
history.

`ptsi_registry` is the persistent-source catalog the architecture puts in
Layer 4 and Phase-1 deliverable #2 calls the "Persistent vs. Transient
Anomaly Segregator". Nothing in M1-M4 built it, and without it the
brief's own example output — "persistent for 27 days, currently 4.2x its
normal FRP" — is not expressible.
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
    Text,
)

from app.db.models import Base  # M1's declarative base — shared MetaData


class PTSIRegistry(Base):
    """What normal looks like for one thermal source, and whether it is
    currently behaving that way."""

    __tablename__ = "ptsi_registry"

    cluster_id = Column(Integer, ForeignKey("clusters.id"), primary_key=True)
    ptsi_score = Column(Float, nullable=True)           # 0-1
    source_class = Column(String(20), nullable=True)     # persistent/intermittent/transient

    # Component scores, stored so a score can be explained rather than
    # just asserted.
    longevity_score = Column(Float, nullable=True)
    reliability_score = Column(Float, nullable=True)
    stability_score = Column(Float, nullable=True)

    # The baseline itself — "its normal".
    baseline_frp_mean = Column(Float, nullable=True)
    baseline_frp_std = Column(Float, nullable=True)
    baseline_frp_median = Column(Float, nullable=True)
    baseline_frp_p95 = Column(Float, nullable=True)
    baseline_n_observations = Column(Integer, nullable=True)

    detection_rate = Column(Float, nullable=True)
    observation_span_days = Column(Float, nullable=True)
    first_seen = Column(DateTime(timezone=True), nullable=True)

    # Current behaviour against that baseline.
    deviation_sigma = Column(Float, nullable=True)
    deviation_multiple = Column(Float, nullable=True)     # the brief's "4.2x"
    is_behaving_normally = Column(Boolean, nullable=True)  # None = no baseline yet
    summary = Column(Text, nullable=True)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class RhythmFingerprintRow(Base):
    """Shift-schedule fingerprint, built on bias-corrected overpass slots."""

    __tablename__ = "rhythm_fingerprints"

    cluster_id = Column(Integer, ForeignKey("clusters.id"), primary_key=True)
    shift_sharpness = Column(Float, nullable=True)
    weekend_suppression = Column(Float, nullable=True)
    day_night_contrast = Column(Float, nullable=True)
    slot_detection_rates = Column(JSON, nullable=True)
    day_of_week_rates = Column(JSON, nullable=True)
    dominant_period_hours = Column(Float, nullable=True)
    periodicity_confidence = Column(Float, nullable=True)
    is_weekly = Column(Boolean, nullable=True)
    schedule_anomaly = Column(Boolean, default=False)
    n_observations = Column(Integer, nullable=True)
    n_expected_passes = Column(Float, nullable=True)
    observation_span_days = Column(Float, nullable=True)
    # Whether the (platform, daynight) grouping really does sit at a
    # consistent local solar hour — a check on the premise of the whole
    # slot model.
    slots_consistent = Column(Boolean, nullable=True)
    reason = Column(Text, nullable=True)
    computed_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class KalmanStateRow(Base):
    """Filtered FRP state and the escalation projection built from it."""

    __tablename__ = "kalman_states"

    cluster_id = Column(Integer, ForeignKey("clusters.id"), primary_key=True)
    frp_estimate = Column(Float, nullable=True)
    frp_rate_estimate = Column(Float, nullable=True)      # MW per hour
    frp_std = Column(Float, nullable=True)
    rate_std = Column(Float, nullable=True)
    covariance = Column(JSON, nullable=True)              # 2x2
    n_updates = Column(Integer, nullable=True)
    mean_nis = Column(Float, nullable=True)               # filter self-consistency

    last_observation_at = Column(DateTime(timezone=True), nullable=True)
    freshness = Column(String(10), nullable=True)          # FRESH/MODERATE/STALE
    age_hours = Column(Float, nullable=True)
    cadence_hours = Column(Float, nullable=True)

    critical_threshold_frp = Column(Float, nullable=True)
    time_to_critical_hours = Column(Float, nullable=True)
    ttc_p50_low = Column(Float, nullable=True)
    ttc_p50_high = Column(Float, nullable=True)
    ttc_p90_low = Column(Float, nullable=True)
    ttc_p90_high = Column(Float, nullable=True)
    probability_reaches_critical = Column(Float, nullable=True)
    escalating = Column(Boolean, default=False)
    already_critical = Column(Boolean, default=False)
    reason = Column(Text, nullable=True)

    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
