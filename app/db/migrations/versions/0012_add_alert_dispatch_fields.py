"""Extend `alerts` for the emergency SMS dispatch feature (test mode).

The table already had `channel` ("sms/email/webhook") but nothing ever
wrote to it — this is the first real writer. Adds the fields an actual
dispatch attempt needs to record: who it went to, whether the provider
(HttpSMS) accepted it, its message id for later status lookups, and the
error text on failure (never silently swallowed — see
app/notifications/sms_client.py).

Revision ID: 0012
Revises: 0011
Create Date: 2026-09-21
"""
from alembic import op
import sqlalchemy as sa

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("alerts", sa.Column("recipient", sa.String(32), nullable=True))
    op.add_column("alerts", sa.Column("status", sa.String(20), nullable=True))  # sent/failed
    op.add_column("alerts", sa.Column("provider", sa.String(20), nullable=True))
    op.add_column("alerts", sa.Column("provider_message_id", sa.String(64), nullable=True))
    op.add_column("alerts", sa.Column("error", sa.String, nullable=True))
    op.create_index("idx_alerts_cluster_channel", "alerts", ["cluster_id", "channel"])


def downgrade() -> None:
    op.drop_index("idx_alerts_cluster_channel", table_name="alerts")
    op.drop_column("alerts", "error")
    op.drop_column("alerts", "provider_message_id")
    op.drop_column("alerts", "provider")
    op.drop_column("alerts", "status")
    op.drop_column("alerts", "recipient")
