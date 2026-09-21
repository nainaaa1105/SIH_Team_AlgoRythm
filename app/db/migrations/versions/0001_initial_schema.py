"""initial schema: hotspots, clusters, facilities, fingerprints, alerts

Revision ID: 0001
Revises:
Create Date: 2026-09-11
"""
from alembic import op
import sqlalchemy as sa
from geoalchemy2 import Geometry

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _timescaledb_available() -> bool:
    """Is the TimescaleDB extension installable on this server?

    TimescaleDB ships no current Windows build, and the project has to be
    runnable on a plain PostgreSQL+PostGIS install. The hypertable below
    is a pure scaling optimisation — time-partitioning of `hotspots` —
    not a semantic requirement: every query, index and constraint in this
    schema behaves identically on a regular table. So probe for it and
    degrade instead of failing the whole migration.

    Probed via pg_available_extensions rather than a try/except around
    CREATE EXTENSION, because a failed statement aborts the surrounding
    transaction in PostgreSQL and would take the rest of the migration
    down with it.
    """
    bind = op.get_bind()
    return bool(
        bind.execute(
            sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'")
        ).scalar()
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    timescale = _timescaledb_available()
    if timescale:
        op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    op.create_table(
        "clusters",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("centroid", Geometry(geometry_type="POINT", srid=4326, spatial_index=False), nullable=False),
        sa.Column("first_seen", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=True),
        sa.Column("n_detections", sa.Integer, server_default="0"),
        sa.Column("cloud_fraction", sa.Float, nullable=True),
        sa.Column("optical_available", sa.Boolean, server_default=sa.false()),
        sa.Column("status", sa.String(20), server_default="active"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_clusters_centroid", "clusters", ["centroid"], postgresql_using="gist")

    op.create_table(
        "facilities",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("facility_type", sa.String(50), nullable=True),
        sa.Column("source", sa.String(20), nullable=True),
        sa.Column("geom", Geometry(geometry_type="GEOMETRY", srid=4326, spatial_index=False), nullable=False),
        sa.Column("prior_weight", sa.Float, server_default="1.0"),
        sa.Column("metadata", sa.JSON, nullable=True),
    )
    op.create_index("idx_facilities_geom", "facilities", ["geom"], postgresql_using="gist")

    op.create_table(
        "hotspots",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("geom", Geometry(geometry_type="POINT", srid=4326, spatial_index=False), nullable=False),
        sa.Column("acq_datetime", sa.DateTime(timezone=True), nullable=False),
        sa.Column("frp", sa.Float, nullable=True),
        sa.Column("brightness", sa.Float, nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("daynight", sa.String(1), nullable=True),
        sa.Column("scan", sa.Float, nullable=True),
        sa.Column("track", sa.Float, nullable=True),
        sa.Column("raw_payload", sa.JSON, nullable=True),
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("source", "geom", "acq_datetime", name="uq_hotspot_identity"),
        sa.CheckConstraint("daynight in ('D','N') OR daynight IS NULL", name="ck_daynight"),
    )
    op.create_index("idx_hotspots_geom", "hotspots", ["geom"], postgresql_using="gist")
    op.create_index("idx_hotspots_cluster_id", "hotspots", ["cluster_id"])
    op.create_index("idx_hotspots_acq_datetime", "hotspots", ["acq_datetime"])

    # TimescaleDB hypertable — must be created after the table exists and
    # requires the partitioning column to be part of every unique constraint,
    # which acq_datetime already is above.
    if timescale:
        op.execute("SELECT create_hypertable('hotspots', 'acq_datetime', if_not_exists => TRUE)")

    op.create_table(
        "fingerprints",
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), primary_key=True),
        sa.Column("pct_cropland", sa.Float, nullable=True),
        sa.Column("pct_forest", sa.Float, nullable=True),
        sa.Column("pct_urban", sa.Float, nullable=True),
        sa.Column("nearest_facility_id", sa.Integer, sa.ForeignKey("facilities.id"), nullable=True),
        sa.Column("facility_distance_m", sa.Float, nullable=True),
        sa.Column("population_density", sa.Float, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("cluster_id", sa.Integer, sa.ForeignKey("clusters.id"), nullable=False),
        sa.Column("severity", sa.String(20), nullable=True),
        sa.Column("message", sa.String, nullable=True),
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("channel", sa.String(20), nullable=True),
    )
    op.create_index("idx_alerts_cluster_id", "alerts", ["cluster_id"])


def downgrade() -> None:
    op.drop_table("alerts")
    op.drop_table("fingerprints")
    op.drop_index("idx_hotspots_acq_datetime", table_name="hotspots")
    op.drop_index("idx_hotspots_cluster_id", table_name="hotspots")
    op.drop_index("idx_hotspots_geom", table_name="hotspots")
    op.drop_table("hotspots")
    op.drop_index("idx_facilities_geom", table_name="facilities")
    op.drop_table("facilities")
    op.drop_index("idx_clusters_centroid", table_name="clusters")
    op.drop_table("clusters")
