"""Role-based signup/login fields on users

Adds the fields the two-role login screen needs: which portal the
account works on ("admin" sees all-India data, "state" is scoped to one
state), plus the identity fields the signup form collects (full name,
government ID, and the represented state for a state account).

Existing rows (there are none in a fresh install, but any created before
this migration during dev/testing) get a safe default so the column can
be NOT NULL going forward: role defaults to "admin" since a state
account without a state is meaningless.

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-19
"""
from alembic import op
import sqlalchemy as sa

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("full_name", sa.String(120), nullable=False, server_default="Unknown"))
    op.add_column("users", sa.Column("government_id", sa.String(64), nullable=False, server_default="UNKNOWN"))
    op.add_column("users", sa.Column("role", sa.String(20), nullable=False, server_default="admin"))
    op.add_column("users", sa.Column("state", sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "state")
    op.drop_column("users", "role")
    op.drop_column("users", "government_id")
    op.drop_column("users", "full_name")
