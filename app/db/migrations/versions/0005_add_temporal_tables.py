"""M5: ptsi_registry, rhythm_fingerprints, kalman_states

Fifth and final link in the shared chain (M1 0001 -> M2 0002 -> M3 0003
-> M4 0004 -> M5 0005). One database, one alembic_version table.

`ptsi_registry` is the persistent-source catalog from architecture Layer
4 / Phase-1 deliverable #2, which no earlier member built.

M5's three feature columns (shift_sharpness, weekend_suppression,
kalman_time_to_critical) already exist on `cluster_features` from 0002 —
M2 reserved them.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-12
"""
from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ptsi_registry",
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), primary_key=True),
        sa.Column("ptsi_score", sa.Float, nullable=True),
        sa.Column("source_class", sa.String(20), nullable=True),
        sa.Column("longevity_score", sa.Float, nullable=True),
        sa.Column("reliability_score", sa.Float, nullable=True),
        sa.Column("stability_score", sa.Float, nullable=True),
        sa.Column("baseline_frp_mean", sa.Float, nullable=True),
        sa.Column("baseline_frp_std", sa.Float, nullable=True),
        sa.Column("baseline_frp_median", sa.Float, nullable=True),
        sa.Column("baseline_frp_p95", sa.Float, nullable=True),
        sa.Column("baseline_n_observations", sa.Integer, nullable=True),
        sa.Column("detection_rate", sa.Float, nullable=True),
        sa.Column("observation_span_days", sa.Float, nullable=True),
        sa.Column("first_seen", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deviation_sigma", sa.Float, nullable=True),
        sa.Column("deviation_multiple", sa.Float, nullable=True),
        sa.Column("is_behaving_normally", sa.Boolean, nullable=True),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_ptsi_source_class", "ptsi_registry", ["source_class"])
    op.create_index("idx_ptsi_score", "ptsi_registry", ["ptsi_score"])

    op.create_table(
        "rhythm_fingerprints",
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), primary_key=True),
        sa.Column("shift_sharpness", sa.Float, nullable=True),
        sa.Column("weekend_suppression", sa.Float, nullable=True),
        sa.Column("day_night_contrast", sa.Float, nullable=True),
        sa.Column("slot_detection_rates", sa.JSON, nullable=True),
        sa.Column("day_of_week_rates", sa.JSON, nullable=True),
        sa.Column("dominant_period_hours", sa.Float, nullable=True),
        sa.Column("periodicity_confidence", sa.Float, nullable=True),
        sa.Column("is_weekly", sa.Boolean, nullable=True),
        sa.Column("schedule_anomaly", sa.Boolean, server_default=sa.false()),
        sa.Column("n_observations", sa.Integer, nullable=True),
        sa.Column("n_expected_passes", sa.Float, nullable=True),
        sa.Column("observation_span_days", sa.Float, nullable=True),
        sa.Column("slots_consistent", sa.Boolean, nullable=True),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "kalman_states",
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), primary_key=True),
        sa.Column("frp_estimate", sa.Float, nullable=True),
        sa.Column("frp_rate_estimate", sa.Float, nullable=True),
        sa.Column("frp_std", sa.Float, nullable=True),
        sa.Column("rate_std", sa.Float, nullable=True),
        sa.Column("covariance", sa.JSON, nullable=True),
        sa.Column("n_updates", sa.Integer, nullable=True),
        sa.Column("mean_nis", sa.Float, nullable=True),
        sa.Column("last_observation_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("freshness", sa.String(10), nullable=True),
        sa.Column("age_hours", sa.Float, nullable=True),
        sa.Column("cadence_hours", sa.Float, nullable=True),
        sa.Column("critical_threshold_frp", sa.Float, nullable=True),
        sa.Column("time_to_critical_hours", sa.Float, nullable=True),
        sa.Column("ttc_p50_low", sa.Float, nullable=True),
        sa.Column("ttc_p50_high", sa.Float, nullable=True),
        sa.Column("ttc_p90_low", sa.Float, nullable=True),
        sa.Column("ttc_p90_high", sa.Float, nullable=True),
        sa.Column("probability_reaches_critical", sa.Float, nullable=True),
        sa.Column("escalating", sa.Boolean, server_default=sa.false()),
        sa.Column("already_critical", sa.Boolean, server_default=sa.false()),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_kalman_escalating", "kalman_states", ["escalating"])


def downgrade() -> None:
    op.drop_index("idx_kalman_escalating", table_name="kalman_states")
    op.drop_table("kalman_states")
    op.drop_table("rhythm_fingerprints")
    op.drop_index("idx_ptsi_score", table_name="ptsi_registry")
    op.drop_index("idx_ptsi_source_class", table_name="ptsi_registry")
    op.drop_table("ptsi_registry")
