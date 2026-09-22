"""M2: cluster_features, classifications, training_labels

Added by Member 2 into Member 1's migration chain deliberately — there is
one shared database and one Alembic version history for the whole team,
so a second chain would produce conflicting `alembic_version` states.

`cluster_features` is the shared 28-feature matrix: M2 creates it and
owns most columns, while M3 (threat_corridor_present), M4 (imagery
columns) and M5 (rhythm columns) UPDATE their own columns on the same
row from their own Celery tasks.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-11
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cluster_features",
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), primary_key=True),
        # CONTEXT (snapshotted from M1's fingerprints)
        sa.Column("pct_cropland", sa.Float, nullable=True),
        sa.Column("pct_forest", sa.Float, nullable=True),
        sa.Column("pct_urban", sa.Float, nullable=True),
        sa.Column("facility_distance_m", sa.Float, nullable=True),
        sa.Column("facility_prior_weight", sa.Float, nullable=True),
        sa.Column("facility_type", sa.String(50), nullable=True),
        sa.Column("population_density", sa.Float, nullable=True),
        sa.Column("near_facility", sa.Float, nullable=True),
        # THERMAL (M2)
        sa.Column("frp_mean", sa.Float, nullable=True),
        sa.Column("frp_max", sa.Float, nullable=True),
        sa.Column("frp_std", sa.Float, nullable=True),
        sa.Column("frp_zscore", sa.Float, nullable=True),
        sa.Column("brightness_mean", sa.Float, nullable=True),
        sa.Column("brightness_max", sa.Float, nullable=True),
        sa.Column("confidence_mean", sa.Float, nullable=True),
        # TEMPORAL (M2)
        sa.Column("persistence_days", sa.Float, nullable=True),
        sa.Column("n_detections", sa.Float, nullable=True),
        sa.Column("detections_per_day", sa.Float, nullable=True),
        sa.Column("night_fraction", sa.Float, nullable=True),
        sa.Column("source_diversity", sa.Float, nullable=True),
        # SPATIAL (M2)
        sa.Column("spatial_extent_km", sa.Float, nullable=True),
        sa.Column("spatial_growth_rate", sa.Float, nullable=True),
        # IMAGERY (M4)
        sa.Column("dozier_temp", sa.Float, nullable=True),
        sa.Column("ndvi", sa.Float, nullable=True),
        sa.Column("ndbi", sa.Float, nullable=True),
        sa.Column("smoke_red_blue_ratio", sa.Float, nullable=True),
        # RHYTHM (M5)
        sa.Column("shift_sharpness", sa.Float, nullable=True),
        sa.Column("weekend_suppression", sa.Float, nullable=True),
        sa.Column("kalman_time_to_critical", sa.Float, nullable=True),
        # M3 — stored for dashboard/alerting, excluded from the model matrix
        sa.Column("threat_corridor_present", sa.Boolean, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "classifications",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), nullable=False),
        sa.Column("predicted_class", sa.String(30), nullable=True),
        sa.Column("class_probabilities", sa.JSON, nullable=True),
        sa.Column("xgb_probabilities", sa.JSON, nullable=True),
        sa.Column("image_probabilities", sa.JSON, nullable=True),
        sa.Column("fusion_weight_xgb", sa.Float, nullable=True),
        sa.Column("confidence_score", sa.Float, nullable=True),
        sa.Column("shap_values", sa.JSON, nullable=True),
        sa.Column("explanation_method", sa.String(40), nullable=True),
        sa.Column("reasons", sa.JSON, nullable=True),
        sa.Column("evidence_weight_notes", sa.JSON, nullable=True),
        sa.Column("model_version", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("cluster_id", name="uq_classification_cluster"),
    )
    op.create_index("idx_classifications_cluster_id", "classifications", ["cluster_id"])
    op.create_index("idx_classifications_predicted_class", "classifications", ["predicted_class"])

    op.create_table(
        "training_labels",
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), primary_key=True),
        sa.Column("label", sa.String(30), nullable=False),
        sa.Column("label_source", sa.String(30), nullable=True),
        sa.Column("label_confidence", sa.Float, nullable=True),
        sa.Column("reason", sa.String, nullable=True),
        sa.Column("is_ambiguous", sa.Boolean, server_default=sa.false()),
        sa.Column("verified_by_analyst", sa.Boolean, server_default=sa.false()),
        sa.Column("labeled_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_training_labels_label", "training_labels", ["label"])


def downgrade() -> None:
    op.drop_index("idx_training_labels_label", table_name="training_labels")
    op.drop_table("training_labels")
    op.drop_index("idx_classifications_predicted_class", table_name="classifications")
    op.drop_index("idx_classifications_cluster_id", table_name="classifications")
    op.drop_table("classifications")
    op.drop_table("cluster_features")
