"""Crown-fire FRP threshold detection column on clusters

Independent of the WUI columns added in 0006: an FRP spike (>=100 MW, or
climbing >=40 MW/h) is diagnostic of a fire transitioning into the
canopy regardless of class or proximity to a settlement. See
geospatial/crown_fire.py for the full mechanism.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-19
"""
from alembic import op
import sqlalchemy as sa

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("clusters", sa.Column("is_crown_fire", sa.Boolean, server_default=sa.false()))


def downgrade() -> None:
    op.drop_column("clusters", "is_crown_fire")
