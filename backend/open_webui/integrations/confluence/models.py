import time
import uuid

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import JSON, BigInteger, Boolean, Column, ForeignKey, Index, Text, UniqueConstraint, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from open_webui.internal.db import Base, get_async_db_context

DEFAULT_INCREMENTAL_CRON = '*/15 * * * *'
DEFAULT_FULL_CRON = '0 2 * * *'
DEFAULT_TIMEZONE = 'Europe/Moscow'
DEFAULT_COLLECTION = 'confluence_minilm_v1'
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_EMBEDDING_VERSION = 'paraphrase-multilingual-minilm-l12-v2-fastembed-0.8.0'
DEFAULT_CHUNKING_VERSION = 'heading-96-overlap16-prefix32-v1'


class ConfluenceConnection(Base):
    __tablename__ = 'confluence_connection'

    id = Column(Text, primary_key=True)
    knowledge_id = Column(Text, ForeignKey('knowledge.id', ondelete='SET NULL'), nullable=True, unique=True)
    name = Column(Text, nullable=False)
    enabled = Column(Boolean, nullable=False, default=False)
    spaces = Column(JSON, nullable=False, default=list)
    collection_name = Column(Text, nullable=False, default=DEFAULT_COLLECTION)
    embedding_model = Column(Text, nullable=False, default=DEFAULT_EMBEDDING_MODEL)
    embedding_dim = Column(BigInteger, nullable=False, default=DEFAULT_EMBEDDING_DIM)
    chunking_version = Column(Text, nullable=False, default=DEFAULT_CHUNKING_VERSION)
    incremental_cron = Column(Text, nullable=False, default=DEFAULT_INCREMENTAL_CRON)
    full_cron = Column(Text, nullable=False, default=DEFAULT_FULL_CRON)
    timezone = Column(Text, nullable=False, default=DEFAULT_TIMEZONE)
    last_watermark = Column(Text, nullable=True)
    status = Column(Text, nullable=False, default='disabled')
    last_error_code = Column(Text, nullable=True)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)


class ConfluencePage(Base):
    __tablename__ = 'confluence_page'

    id = Column(Text, primary_key=True)
    connection_id = Column(Text, ForeignKey('confluence_connection.id', ondelete='CASCADE'), nullable=False)
    page_id = Column(Text, nullable=False)
    space = Column(Text, nullable=False)
    title = Column(Text, nullable=False)
    url = Column(Text, nullable=True)
    active_version = Column(Text, nullable=True)
    active_hash = Column(Text, nullable=True)
    available = Column(Boolean, nullable=False, default=True)
    restrictions_checked_at = Column(BigInteger, nullable=True)
    last_seen_at = Column(BigInteger, nullable=True)
    page_metadata = Column('metadata', JSON, nullable=False, default=dict)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)

    __table_args__ = (
        UniqueConstraint('connection_id', 'page_id', name='uq_confluence_page_connection_page'),
        Index('ix_confluence_page_connection_space', 'connection_id', 'space'),
        Index('ix_confluence_page_active', 'connection_id', 'page_id', 'active_version', 'active_hash'),
    )


class ConfluenceRun(Base):
    __tablename__ = 'confluence_run'

    id = Column(Text, primary_key=True)
    connection_id = Column(Text, ForeignKey('confluence_connection.id', ondelete='CASCADE'), nullable=False)
    mode = Column(Text, nullable=False)
    status = Column(Text, nullable=False, default='queued')
    owner_id = Column(Text, nullable=True)
    lease_expires_at = Column(BigInteger, nullable=True)
    heartbeat_at = Column(BigInteger, nullable=True)
    started_at = Column(BigInteger, nullable=True)
    finished_at = Column(BigInteger, nullable=True)
    watermark = Column(Text, nullable=True)
    counters = Column(JSON, nullable=False, default=dict)
    safe_error_code = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)

    __table_args__ = (
        Index('ix_confluence_run_connection_status', 'connection_id', 'status'),
        Index('ix_confluence_run_lease', 'lease_expires_at'),
    )


class ConfluenceConnectionForm(BaseModel):
    enabled: bool = False
    spaces: list[str] = Field(default_factory=list)
    incremental_cron: str = DEFAULT_INCREMENTAL_CRON
    full_cron: str = DEFAULT_FULL_CRON
    timezone: str = DEFAULT_TIMEZONE
    collection_name: str = DEFAULT_COLLECTION


class ConfluenceSearchForm(BaseModel):
    query: str
    count: int = 8


class ConfluenceSyncForm(BaseModel):
    mode: str = 'incremental'


class ConfluenceConnectionModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    knowledge_id: str | None = None
    name: str
    enabled: bool
    spaces: list[str]
    collection_name: str
    embedding_model: str
    embedding_dim: int
    chunking_version: str
    incremental_cron: str
    full_cron: str
    timezone: str
    last_watermark: str | None = None
    status: str
    last_error_code: str | None = None
    created_at: int
    updated_at: int


class ConfluenceRunModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    connection_id: str
    mode: str
    status: str
    owner_id: str | None = None
    lease_expires_at: int | None = None
    heartbeat_at: int | None = None
    started_at: int | None = None
    finished_at: int | None = None
    watermark: str | None = None
    counters: dict = Field(default_factory=dict)
    safe_error_code: str | None = None
    error: str | None = None
    created_at: int
    updated_at: int


class ConfluenceConnectionTable:
    async def get_default(self, db: AsyncSession | None = None) -> ConfluenceConnectionModel | None:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(ConfluenceConnection).order_by(ConfluenceConnection.created_at.asc()))
            connection = result.scalars().first()
            return ConfluenceConnectionModel.model_validate(connection) if connection else None

    async def upsert_default(
        self,
        form_data: ConfluenceConnectionForm,
        knowledge_id: str | None = None,
        db: AsyncSession | None = None,
    ) -> ConfluenceConnectionModel:
        async with get_async_db_context(db) as db:
            now = int(time.time())
            result = await db.execute(select(ConfluenceConnection).order_by(ConfluenceConnection.created_at.asc()))
            connection = result.scalars().first()
            values = {
                'enabled': form_data.enabled,
                'spaces': form_data.spaces,
                'incremental_cron': form_data.incremental_cron,
                'full_cron': form_data.full_cron,
                'timezone': form_data.timezone,
                'collection_name': form_data.collection_name,
                'status': 'enabled' if form_data.enabled else 'disabled',
                'updated_at': now,
            }
            if knowledge_id:
                values['knowledge_id'] = knowledge_id

            if connection:
                await db.execute(update(ConfluenceConnection).filter_by(id=connection.id).values(**values))
                await db.commit()
                result = await db.execute(select(ConfluenceConnection).filter_by(id=connection.id))
                connection = result.scalars().first()
            else:
                connection = ConfluenceConnection(
                    id=str(uuid.uuid4()),
                    knowledge_id=knowledge_id,
                    name='Confluence',
                    created_at=now,
                    **values,
                )
                db.add(connection)
                await db.commit()
                await db.refresh(connection)

            return ConfluenceConnectionModel.model_validate(connection)

    async def queue_run(self, connection_id: str, mode: str, db: AsyncSession | None = None) -> ConfluenceRunModel:
        mode = mode if mode in {'incremental', 'full'} else 'incremental'
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(ConfluenceRun)
                .filter(
                    ConfluenceRun.connection_id == connection_id,
                    ConfluenceRun.mode == mode,
                    ConfluenceRun.status.in_(['queued', 'running']),
                )
                .order_by(ConfluenceRun.created_at.asc())
            )
            run = result.scalars().first()
            if not run:
                now = int(time.time())
                run = ConfluenceRun(
                    id=str(uuid.uuid4()),
                    connection_id=connection_id,
                    mode=mode,
                    status='queued',
                    counters={},
                    created_at=now,
                    updated_at=now,
                )
                db.add(run)
                await db.commit()
                await db.refresh(run)
            return ConfluenceRunModel.model_validate(run)

    async def list_runs(
        self,
        connection_id: str,
        limit: int = 30,
        db: AsyncSession | None = None,
    ) -> list[ConfluenceRunModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(ConfluenceRun)
                .filter_by(connection_id=connection_id)
                .order_by(ConfluenceRun.created_at.desc())
                .limit(limit)
            )
            return [ConfluenceRunModel.model_validate(run) for run in result.scalars().all()]

    async def get_run(
        self,
        run_id: str,
        db: AsyncSession | None = None,
    ) -> ConfluenceRunModel | None:
        async with get_async_db_context(db) as db:
            result = await db.execute(select(ConfluenceRun).filter_by(id=run_id))
            run = result.scalars().first()
            return ConfluenceRunModel.model_validate(run) if run else None

    async def acquire_next_run(
        self,
        owner_id: str,
        lease_seconds: int = 300,
        db: AsyncSession | None = None,
    ) -> ConfluenceRunModel | None:
        async with get_async_db_context(db) as db:
            now = int(time.time())
            result = await db.execute(
                select(ConfluenceRun)
                .filter(
                    or_(
                        ConfluenceRun.status == 'queued',
                        (ConfluenceRun.status == 'running') & (ConfluenceRun.lease_expires_at < now),
                    )
                )
                .order_by(ConfluenceRun.created_at.asc())
                .limit(1)
            )
            run = result.scalars().first()
            if not run:
                return None

            updated = await db.execute(
                update(ConfluenceRun)
                .filter(
                    ConfluenceRun.id == run.id,
                    or_(
                        ConfluenceRun.status == 'queued',
                        (ConfluenceRun.status == 'running') & (ConfluenceRun.lease_expires_at < now),
                    ),
                )
                .values(
                    status='running',
                    owner_id=owner_id,
                    lease_expires_at=now + lease_seconds,
                    heartbeat_at=now,
                    started_at=run.started_at or now,
                    updated_at=now,
                )
            )
            await db.commit()
            if updated.rowcount != 1:
                return None

            result = await db.execute(select(ConfluenceRun).filter_by(id=run.id))
            run = result.scalars().first()
            return ConfluenceRunModel.model_validate(run) if run else None

    async def heartbeat_run(
        self,
        run_id: str,
        owner_id: str,
        lease_seconds: int = 300,
        db: AsyncSession | None = None,
    ) -> bool:
        async with get_async_db_context(db) as db:
            now = int(time.time())
            result = await db.execute(
                update(ConfluenceRun)
                .filter_by(id=run_id, owner_id=owner_id, status='running')
                .values(heartbeat_at=now, lease_expires_at=now + lease_seconds, updated_at=now)
            )
            await db.commit()
            return result.rowcount == 1

    async def finish_run(
        self,
        run_id: str,
        owner_id: str,
        status: str,
        counters: dict | None = None,
        safe_error_code: str | None = None,
        error: str | None = None,
        db: AsyncSession | None = None,
    ) -> bool:
        async with get_async_db_context(db) as db:
            now = int(time.time())
            result = await db.execute(
                update(ConfluenceRun)
                .filter_by(id=run_id, owner_id=owner_id, status='running')
                .values(
                    status=status,
                    counters=counters or {},
                    safe_error_code=safe_error_code,
                    error=error,
                    finished_at=now,
                    lease_expires_at=None,
                    updated_at=now,
                )
            )
            await db.commit()
            return result.rowcount == 1

    async def update_state(
        self,
        connection_id: str,
        status: str,
        *,
        watermark: str | None = None,
        last_error_code: str | None = None,
        db: AsyncSession | None = None,
    ) -> None:
        async with get_async_db_context(db) as db:
            values = {
                'status': status,
                'last_error_code': last_error_code,
                'updated_at': int(time.time()),
            }
            if watermark is not None:
                values['last_watermark'] = watermark
            await db.execute(update(ConfluenceConnection).filter_by(id=connection_id).values(**values))
            await db.commit()

    async def sync_pages(
        self,
        connection_id: str,
        pages: list[dict],
        db: AsyncSession | None = None,
    ) -> dict[str, int]:
        async with get_async_db_context(db) as db:
            now = int(time.time())
            result = await db.execute(select(ConfluencePage).filter_by(connection_id=connection_id))
            existing = {page.page_id: page for page in result.scalars().all()}
            seen: set[str] = set()
            created = 0
            updated_count = 0

            for item in pages:
                page_id = str(item.get('page_id') or '').strip()
                version = str(item.get('page_version') or '').strip()
                content_hash = str(item.get('content_hash') or '').strip()
                if not page_id or not version or not content_hash:
                    continue
                seen.add(page_id)
                values = {
                    'space': str(item.get('space_key') or ''),
                    'title': str(item.get('title') or ''),
                    'url': str(item.get('url') or '') or None,
                    'active_version': version,
                    'active_hash': content_hash,
                    'available': True,
                    'restrictions_checked_at': now,
                    'last_seen_at': now,
                    'page_metadata': {
                        'updated_at': item.get('updated_at'),
                        'embedding_version': item.get('embedding_version'),
                        'chunker_version': item.get('chunker_version'),
                    },
                    'updated_at': now,
                }
                page = existing.get(page_id)
                if page:
                    for key, value in values.items():
                        setattr(page, key, value)
                    updated_count += 1
                else:
                    db.add(
                        ConfluencePage(
                            id=str(uuid.uuid4()),
                            connection_id=connection_id,
                            page_id=page_id,
                            created_at=now,
                            **values,
                        )
                    )
                    created += 1

            unavailable = 0
            for page_id, page in existing.items():
                if page_id not in seen and page.available:
                    page.available = False
                    page.updated_at = now
                    unavailable += 1

            await db.commit()
            return {
                'pages_created': created,
                'pages_updated': updated_count,
                'pages_unavailable': unavailable,
                'pages_active': len(seen),
            }


ConfluenceConnections = ConfluenceConnectionTable()
