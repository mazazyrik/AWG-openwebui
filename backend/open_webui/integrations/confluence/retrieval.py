import asyncio
import logging
import os
import time

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from open_webui.integrations.confluence.client import QDRANT_API_KEY_ENV, QDRANT_URL_ENV, ConfluenceMCPClient
from open_webui.integrations.confluence.models import (
    DEFAULT_EMBEDDING_VERSION,
    ConfluenceConnections,
    ConfluencePage,
)
from open_webui.internal.db import get_async_db_context
from open_webui.models.knowledge import KnowledgeModel

log = logging.getLogger(__name__)


_embedding_model = None


def _embed_query(query: str, model_name: str) -> list[float]:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model.encode(query, normalize_embeddings=True).tolist()


async def _active_page(connection_id: str, page_id: str, version: str | None, content_hash: str | None):
    if not page_id or not version or not content_hash:
        return None
    async with get_async_db_context() as db:
        result = await db.execute(
            select(ConfluencePage)
            .filter(
                ConfluencePage.connection_id == connection_id,
                ConfluencePage.page_id == str(page_id),
                ConfluencePage.active_version == str(version),
                ConfluencePage.active_hash == str(content_hash),
                ConfluencePage.available.is_(True),
            )
            .limit(1)
        )
        return result.scalars().first()


def _payload_from_point(point) -> tuple[str, dict, float]:
    data = point.model_dump() if hasattr(point, 'model_dump') else dict(point)
    payload = data.get('payload') or {}
    metadata = payload.get('metadata') or payload
    content = payload.get('text') or payload.get('content') or metadata.get('text') or ''
    score = data.get('score') or data.get('distance') or 0
    return content, metadata, score


def _search_points(query: str, connection, qdrant_url: str, count: int):
    vector = _embed_query(query, connection.embedding_model)
    qdrant = QdrantClient(
        url=qdrant_url,
        api_key=os.getenv(QDRANT_API_KEY_ENV),
        timeout=30,
    )
    try:
        return qdrant.query_points(
            collection_name=connection.collection_name,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key='source',
                        match=models.MatchValue(value='confluence'),
                    ),
                    models.FieldCondition(
                        key='embedding_version',
                        match=models.MatchValue(value=DEFAULT_EMBEDDING_VERSION),
                    ),
                ]
            ),
            limit=max(count * 3, count),
            with_payload=True,
            with_vectors=False,
        ).points
    finally:
        qdrant.close()


async def _chunk_from_point(
    point,
    connection,
    knowledge: KnowledgeModel,
    client: ConfluenceMCPClient,
    spaces: set[str],
    seen: set[tuple],
    page_counts: dict[str, int],
) -> dict | None:
    content, metadata, score = _payload_from_point(point)
    page_id = str(metadata.get('page_id') or metadata.get('confluence_page_id') or '')
    version = str(metadata.get('version') or metadata.get('page_version') or '')
    content_hash = str(metadata.get('hash') or metadata.get('content_hash') or '')
    space = str(metadata.get('space') or metadata.get('space_key') or '')
    key = (page_id, version, content_hash, content[:80])
    if key in seen or (spaces and space not in spaces) or page_counts.get(page_id, 0) >= 2:
        return None
    page = await _active_page(connection.id, page_id, version, content_hash)
    if not page or not await client.can_read_page(page_id):
        return None

    seen.add(key)
    page_counts[page_id] = page_counts.get(page_id, 0) + 1
    result_metadata = {
        **metadata,
        'name': page.title,
        'source': page.title,
        'title': page.title,
        'url': page.url,
        'file_id': f'confluence-{page_id}',
        'knowledge_id': knowledge.id,
        'knowledge_name': knowledge.name,
        'provider': 'confluence',
        'external': True,
        'page_id': page_id,
        'version': version,
        'space': space,
        'hash': content_hash,
    }
    return {'content': content, 'metadata': result_metadata, 'distance': score}


async def retrieve_confluence_knowledge(
    knowledge: KnowledgeModel,
    queries: list[str],
    count: int,
    user=None,
) -> dict:
    connection = await ConfluenceConnections.get_default()
    if not connection or not connection.enabled:
        raise RuntimeError('Confluence knowledge connection is disabled')
    if connection.knowledge_id != knowledge.id:
        raise RuntimeError('Knowledge is not bound to the Confluence connection')

    qdrant_url = os.getenv(QDRANT_URL_ENV)
    if not qdrant_url:
        raise RuntimeError('Confluence Qdrant endpoint is not configured')

    started_at = time.monotonic()
    client = ConfluenceMCPClient()
    chunks = []
    seen = set()
    page_counts: dict[str, int] = {}
    spaces = set(connection.spaces or [])

    for query in queries:
        points = await asyncio.to_thread(_search_points, query, connection, qdrant_url, count)
        for point in points:
            chunk = await _chunk_from_point(
                point,
                connection,
                knowledge,
                client,
                spaces,
                seen,
                page_counts,
            )
            if chunk:
                chunks.append(chunk)
            if len(chunks) >= min(count, 8):
                break
        if len(chunks) >= min(count, 8):
            break

    log.info(
        'confluence_retrieval knowledge_id=%s user_id=%s latency_ms=%s result_count=%s',
        knowledge.id,
        getattr(user, 'id', None),
        round((time.monotonic() - started_at) * 1000),
        len(chunks),
    )
    return {
        'documents': [[chunk['content'] for chunk in chunks]],
        'metadatas': [[chunk['metadata'] for chunk in chunks]],
        'distances': [[chunk['distance'] for chunk in chunks]],
    }
