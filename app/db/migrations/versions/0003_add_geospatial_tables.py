"""M3: facility_attributions, plumes, threat_corridors, evacuation_routes

Added by Member 3 into the shared migration chain (M1 0001 -> M2 0002 ->
M3 0003). One database, one alembic_version table.

M3's outputs are geometries, so they need their own tables rather than
scalar columns on `cluster_features` — the single scalar M3 owns there
is `threat_corridor_present`, created back in 0002.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-12
"""
from alembic import op
import sqlalchemy as sa
from geoalchemy2 import Geometry

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "facility_attributions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), nullable=False),
        sa.Column("facility_id", sa.Integer, sa.ForeignKey("facilities.id"), nullable=False),
        sa.Column("distance_m", sa.Float, nullable=True),
        sa.Column("inside_polygon", sa.Boolean, server_default=sa.false()),
        sa.Column("attribution_probability", sa.Float, nullable=True),
        sa.Column("rank", sa.Integer, nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("cluster_id", "facility_id", name="uq_attribution_cluster_facility"),
    )
    op.create_index("idx_attributions_cluster_id", "facility_attributions", ["cluster_id"])

    op.create_table(
        "plumes",
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), primary_key=True),
        sa.Column("geom", Geometry(geometry_type="POLYGON", srid=4326, spatial_index=False), nullable=True),
        sa.Column("wind_speed_ms", sa.Float, nullable=True),
        sa.Column("wind_direction_deg", sa.Float, nullable=True),
        sa.Column("downwind_bearing_deg", sa.Float, nullable=True),
        sa.Column("wind_source", sa.String(20), nullable=True),
        sa.Column("stability_class", sa.String(1), nullable=True),
        sa.Column("plume_rise_m", sa.Float, nullable=True),
        sa.Column("length_m", sa.Float, nullable=True),
        sa.Column("area_km2", sa.Float, nullable=True),
        sa.Column("chemical_profile", sa.JSON, nullable=True),
        sa.Column("population_exposed", sa.Integer, nullable=True),
        sa.Column("population_known", sa.Boolean, server_default=sa.false()),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_plumes_geom", "plumes", ["geom"], postgresql_using="gist")

    op.create_table(
        "threat_corridors",
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), primary_key=True),
        sa.Column("geom", Geometry(geometry_type="POLYGON", srid=4326, spatial_index=False), nullable=True),
        sa.Column("bearing_deg", sa.Float, nullable=True),
        sa.Column("half_angle_deg", sa.Float, nullable=True),
        sa.Column("length_km", sa.Float, nullable=True),
        sa.Column("spread_rate_km_day", sa.Float, nullable=True),
        sa.Column("trajectory_confidence", sa.Float, nullable=True),
        sa.Column("projection_hours", sa.Float, nullable=True),
        sa.Column("threatened_facilities", sa.JSON, nullable=True),
        sa.Column("min_time_to_impact_hours", sa.Float, nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_threat_corridors_geom", "threat_corridors", ["geom"], postgresql_using="gist")

    op.create_table(
        "evacuation_routes",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), nullable=False),
        sa.Column("geom", Geometry(geometry_type="LINESTRING", srid=4326), nullable=True),
        sa.Column("origin_name", sa.String(255), nullable=True),
        sa.Column("destination_name", sa.String(255), nullable=True),
        sa.Column("length_km", sa.Float, nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_evacuation_routes_cluster_id", "evacuation_routes", ["cluster_id"])


def downgrade() -> None:
    op.drop_index("idx_evacuation_routes_cluster_id", table_name="evacuation_routes")
    op.drop_table("evacuation_routes")
    op.drop_index("idx_threat_corridors_geom", table_name="threat_corridors")
    op.drop_table("threat_corridors")
    op.drop_index("idx_plumes_geom", table_name="plumes")
    op.drop_table("plumes")
    op.drop_index("idx_attributions_cluster_id", table_name="facility_attributions")
    op.drop_table("facility_attributions")
