import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c4d8e2f1a7b3"
down_revision = "b7f3a1c9d2e4"  # <-- SET THIS to your current head (`alembic heads`)
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("chunk_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chunks", "chunk_metadata")
