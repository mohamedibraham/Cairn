from alembic import op

# revision identifiers, used by Alembic.
revision = "b7f3a1c9d2e4"
down_revision = "a10fbdc77f2a"  # <-- SET THIS to your current head (see `alembic heads`)
branch_labels = None
depends_on = None

# columns that were `timestamp without time zone` -> need type change + default
NAIVE_COLUMNS = [
    ("artifacts", "created_at"),
    ("chunks", "created_at"),
    ("file_processing_events", "created_at"),
    ("file_versions", "created_at"),
    ("message_citations", "created_at"),
    ("messages", "created_at"),
    ("search_queries", "created_at"),
]

# columns that were already timestamptz -> only the default is wrong
TZ_COLUMNS = [
    ("refresh_tokens", "created_at"),
    ("workspace_activity_logs", "created_at"),
    ("workspace_invites", "created_at"),
    ("workspace_join_requests", "created_at"),
    ("workspace_members", "joined_at"),
]


def upgrade() -> None:
    for table, column in NAIVE_COLUMNS:
        # Drop the bogus constant default first so the type change never has to
        # re-cast it, then convert, then install the real default.
        op.execute(f'ALTER TABLE {table} ALTER COLUMN {column} DROP DEFAULT')
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {column} "
            f"TYPE timestamp with time zone USING {column} AT TIME ZONE 'UTC'"
        )
        op.execute(f'ALTER TABLE {table} ALTER COLUMN {column} SET DEFAULT now()')

    for table, column in TZ_COLUMNS:
        op.execute(f'ALTER TABLE {table} ALTER COLUMN {column} SET DEFAULT now()')


def downgrade() -> None:
    # Intentionally a no-op: the previous state was a bug (every row stamped
    # with the table-creation time). Re-creating it would serve no purpose.
    pass
