import os
from typing import Any

from qdrant_client import QdrantClient, models

from open_webui.integrations.confluence.client import (
    QDRANT_API_KEY_ENV,
    QDRANT_URL_ENV,
)
from open_webui.integrations.confluence.models import (
    DEFAULT_EMBEDDING_VERSION,
    ConfluenceConnectionModel,
)


def indexed_pages(connection: ConfluenceConnectionModel) -> list[dict[str, Any]]:
    url = os.getenv(QDRANT_URL_ENV)
    if not url:
        raise RuntimeError('qdrant_not_configured')
    client = QdrantClient(
        url=url,
        api_key=os.getenv(QDRANT_API_KEY_ENV),
        timeout=30,
    )
    pages: dict[str, dict[str, Any]] = {}
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                collection_name=connection.collection_name,
                scroll_filter=models.Filter(
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
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                page_id = str(payload.get('page_id') or '')
                if not page_id:
                    continue
                previous = pages.get(page_id)
                version = int(payload.get('page_version') or 0)
                if previous is None or version >= int(previous.get('page_version') or 0):
                    pages[page_id] = payload
            if offset is None:
                break
    finally:
        client.close()
    return list(pages.values())
