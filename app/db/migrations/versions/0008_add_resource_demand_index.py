"""WFDSS-inspired Resource Demand Index columns on clusters

Independent of the WUI (0006) and crown-fire (0007) columns: `rdi_score`
and `coa_type` are a derived ranking that reuses those columns'
persisted values rather than re-deriving them. See
geospatial/decision_engine.py for the scoring mechanism.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-19
"""
from alembic import op
import sqlalchemy as sa

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("clusters", sa.Column("rdi_score", sa.Float, server_default="0.0"))
    op.add_column(
        "clusters",
        sa.Column("coa_type", sa.String(100), server_default="MONITOR_ECOLOGICAL_BENEFIT"),
    )


def downgrade() -> None:
    op.drop_column("clusters", "coa_type")
    op.drop_column("clusters", "rdi_score")
