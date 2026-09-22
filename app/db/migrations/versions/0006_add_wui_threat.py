"""WUI (wildland-urban interface) proximity columns on clusters

Added by the WUI Proximity Alert feature: how close a spreading fire
(wildfire / agricultural_burning) is to the nearest ESA WorldCover
built-up pixel, and the ember-jump / 12-hour-window priority that
distance implies. See geospatial/wui_analysis.py for the full mechanism.

Scalar columns on `clusters` rather than a new joined table — see the
comment on `Cluster.wui_threat` in app/db/models.py for why that does
not cross the column-ownership boundary `cluster_features` enforces.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-19
"""
from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("clusters", sa.Column("wui_threat", sa.Boolean, server_default=sa.false()))
    op.add_column("clusters", sa.Column("wui_distance_m", sa.Float, nullable=True))
    op.add_column("clusters", sa.Column("wui_eta_hours", sa.Float, nullable=True))
    op.add_column("clusters", sa.Column("wui_bearing_deg", sa.Float, nullable=True))
    op.add_column("clusters", sa.Column("wui_threatened_asset", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("clusters", "wui_threatened_asset")
    op.drop_column("clusters", "wui_bearing_deg")
    op.drop_column("clusters", "wui_eta_hours")
    op.drop_column("clusters", "wui_distance_m")
    op.drop_column("clusters", "wui_threat")
