""" add blocks column in messages 

Revision ID: 7b42172480d6
Revises: c4d8e2f1a7b3
Create Date: 2026-10-19 13:07:32.535652

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "7b42172480d6"
down_revision: Union[str, Sequence[str], None] = "c4d8e2f1a7b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add blocks column to messages table."""
    op.add_column(
        "messages",
        sa.Column(
            "blocks",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Remove blocks column from messages table."""
    op.drop_column("messages", "blocks")

