"""Add Confluence integration tables

Revision ID: 0f2a3b4c5d6e
Revises: f1e2d3c4b5a6
Create Date: 2026-09-14 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from open_webui.migrations.util import get_existing_tables

revision: str = '0f2a3b4c5d6e'
down_revision: str | None = 'd4c1a8e37b62'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    existing_tables = set(get_existing_tables())

    if 'confluence_connection' not in existing_tables:
        op.create_table(
            'confluence_connection',
            sa.Column('id', sa.Text(), nullable=False, primary_key=True),
            sa.Column(
                'knowledge_id',
                sa.Text(),
                sa.ForeignKey('knowledge.id', ondelete='SET NULL'),
                nullable=True,
                unique=True,
            ),
            sa.Column('name', sa.Text(), nullable=False),
            sa.Column('enabled', sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column('spaces', sa.JSON(), nullable=False),
            sa.Column('collection_name', sa.Text(), nullable=False),
            sa.Column('embedding_model', sa.Text(), nullable=False),
            sa.Column('embedding_dim', sa.BigInteger(), nullable=False),
            sa.Column('chunking_version', sa.Text(), nullable=False),
            sa.Column('incremental_cron', sa.Text(), nullable=False),
            sa.Column('full_cron', sa.Text(), nullable=False),
            sa.Column('timezone', sa.Text(), nullable=False),
            sa.Column('last_watermark', sa.Text(), nullable=True),
            sa.Column('status', sa.Text(), nullable=False),
            sa.Column('last_error_code', sa.Text(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.Column('updated_at', sa.BigInteger(), nullable=False),
        )

    if 'confluence_page' not in existing_tables:
        op.create_table(
            'confluence_page',
            sa.Column('id', sa.Text(), nullable=False, primary_key=True),
            sa.Column(
                'connection_id',
                sa.Text(),
                sa.ForeignKey('confluence_connection.id', ondelete='CASCADE'),
                nullable=False,
            ),
            sa.Column('page_id', sa.Text(), nullable=False),
            sa.Column('space', sa.Text(), nullable=False),
            sa.Column('title', sa.Text(), nullable=False),
            sa.Column('url', sa.Text(), nullable=True),
            sa.Column('active_version', sa.Text(), nullable=True),
            sa.Column('active_hash', sa.Text(), nullable=True),
            sa.Column('available', sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column('restrictions_checked_at', sa.BigInteger(), nullable=True),
            sa.Column('last_seen_at', sa.BigInteger(), nullable=True),
            sa.Column('metadata', sa.JSON(), nullable=False),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.Column('updated_at', sa.BigInteger(), nullable=False),
            sa.UniqueConstraint('connection_id', 'page_id', name='uq_confluence_page_connection_page'),
        )
        op.create_index('ix_confluence_page_connection_space', 'confluence_page', ['connection_id', 'space'])
        op.create_index(
            'ix_confluence_page_active',
            'confluence_page',
            ['connection_id', 'page_id', 'active_version', 'active_hash'],
        )

    if 'confluence_run' not in existing_tables:
        op.create_table(
            'confluence_run',
            sa.Column('id', sa.Text(), nullable=False, primary_key=True),
            sa.Column(
                'connection_id',
                sa.Text(),
                sa.ForeignKey('confluence_connection.id', ondelete='CASCADE'),
                nullable=False,
            ),
            sa.Column('mode', sa.Text(), nullable=False),
            sa.Column('status', sa.Text(), nullable=False),
            sa.Column('owner_id', sa.Text(), nullable=True),
            sa.Column('lease_expires_at', sa.BigInteger(), nullable=True),
            sa.Column('heartbeat_at', sa.BigInteger(), nullable=True),
            sa.Column('started_at', sa.BigInteger(), nullable=True),
            sa.Column('finished_at', sa.BigInteger(), nullable=True),
            sa.Column('watermark', sa.Text(), nullable=True),
            sa.Column('counters', sa.JSON(), nullable=False),
            sa.Column('safe_error_code', sa.Text(), nullable=True),
            sa.Column('error', sa.Text(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=False),
            sa.Column('updated_at', sa.BigInteger(), nullable=False),
        )
        op.create_index('ix_confluence_run_connection_status', 'confluence_run', ['connection_id', 'status'])
        op.create_index('ix_confluence_run_lease', 'confluence_run', ['lease_expires_at'])


def downgrade() -> None:
    existing_tables = set(get_existing_tables())
    if 'confluence_run' in existing_tables:
        op.drop_index('ix_confluence_run_lease', table_name='confluence_run')
        op.drop_index('ix_confluence_run_connection_status', table_name='confluence_run')
        op.drop_table('confluence_run')
    if 'confluence_page' in existing_tables:
        op.drop_index('ix_confluence_page_active', table_name='confluence_page')
        op.drop_index('ix_confluence_page_connection_space', table_name='confluence_page')
        op.drop_table('confluence_page')
    if 'confluence_connection' in existing_tables:
        op.drop_table('confluence_connection')
