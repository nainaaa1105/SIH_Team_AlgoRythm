"""Persist the footprint-area figure resource_demand already computes

geospatial.decision_engine.footprint_area_ha was computed fresh on every
request and discarded — the new Fire Suppression tab needs it for every
cluster at once, so geospatial.tasks.resource_demand now persists it
alongside rdi_score/coa_type instead of recomputing it 900+ times per
page load.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-21
"""
from alembic import op
import sqlalchemy as sa

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("clusters", sa.Column("area_ha", sa.Float, nullable=True))


def downgrade() -> None:
    op.drop_column("clusters", "area_ha")
