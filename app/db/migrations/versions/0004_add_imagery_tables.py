"""M4: sentinel2_patches, image_predictions, thermal_retrievals

Fourth link in the shared chain (M1 0001 -> M2 0002 -> M3 0003 ->
M4 0004 -> M5 0005). One database, one alembic_version table.

M4's four feature columns (dozier_temp, ndvi, ndbi,
smoke_red_blue_ratio) already exist on `cluster_features` from 0002 —
M2 reserved them. This migration only adds the tables that hold what
does not fit in a scalar feature column: fetched patches, the image
model's standalone verdict, and the thermal retrieval with its
assumptions.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-12
"""
from alembic import op
import sqlalchemy as sa
from geoalchemy2 import Geometry

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sentinel2_patches",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), nullable=False),
        sa.Column("file_path", sa.Text, nullable=True),
        sa.Column("thumbnail_path", sa.Text, nullable=True),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("bbox", Geometry(geometry_type="POLYGON", srid=4326, spatial_index=False), nullable=True),
        sa.Column("cloud_percentage", sa.Float, nullable=True),
        sa.Column("bands", sa.JSON, nullable=True),
        sa.Column("width_px", sa.Integer, nullable=True),
        sa.Column("height_px", sa.Integer, nullable=True),
        sa.Column("source", sa.String(40), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_s2_patches_cluster_id", "sentinel2_patches", ["cluster_id"])
    op.create_index("idx_s2_patches_bbox", "sentinel2_patches", ["bbox"], postgresql_using="gist")

    op.create_table(
        "image_predictions",
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), primary_key=True),
        sa.Column("patch_id", sa.Integer, sa.ForeignKey("sentinel2_patches.id"), nullable=True),
        sa.Column("class_probabilities", sa.JSON, nullable=True),
        sa.Column("predicted_class", sa.String(30), nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("model_version", sa.String(20), nullable=True),
        sa.Column("smoke_detected", sa.Boolean, server_default=sa.false()),
        sa.Column("smoke_bearing_deg", sa.Float, nullable=True),
        sa.Column("smoke_coverage", sa.Float, nullable=True),
        sa.Column("plume_agreement", sa.JSON, nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "thermal_retrievals",
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), primary_key=True),
        sa.Column("fire_temperature_k", sa.Float, nullable=True),
        sa.Column("fire_fraction", sa.Float, nullable=True),
        sa.Column("fire_area_m2", sa.Float, nullable=True),
        sa.Column("background_temperature_k", sa.Float, nullable=True),
        sa.Column("background_source", sa.String(30), nullable=True),
        sa.Column("sensor", sa.String(20), nullable=True),
        sa.Column("converged", sa.Boolean, server_default=sa.false()),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("n_detections_used", sa.Integer, nullable=True),
        # Uncertainty bracket from perturbing the assumed background —
        # the inversion is ill-conditioned for small hot fires.
        sa.Column("temperature_low_k", sa.Float, nullable=True),
        sa.Column("temperature_high_k", sa.Float, nullable=True),
        sa.Column("well_constrained", sa.Boolean, nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("thermal_retrievals")
    op.drop_table("image_predictions")
    op.drop_index("idx_s2_patches_bbox", table_name="sentinel2_patches")
    op.drop_index("idx_s2_patches_cluster_id", table_name="sentinel2_patches")
    op.drop_table("sentinel2_patches")
