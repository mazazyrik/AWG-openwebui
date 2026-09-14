# Native Confluence integration

The integration delegates Confluence discovery and indexing to the private
`awg-confluence-rag` service. OpenWebUI queues synchronization through its admin
API, mirrors safe page metadata from Qdrant, and checks current read restrictions
through the source MCP before returning a chunk.

Required runtime variables:

- `CONFLUENCE_RAG_API_URL`: internal HTTP URL of `awg-confluence-rag`
- `CONFLUENCE_RAG_ADMIN_TOKEN`: bearer token for its admin API
- `CONFLUENCE_MCP_URL`: private Streamable HTTP MCP URL
- `CONFLUENCE_MCP_TOKEN`: optional MCP bearer token
- `CONFLUENCE_QDRANT_URL`: private Qdrant URL
- `CONFLUENCE_QDRANT_API_KEY`: Qdrant API key

Run the worker as a separate process:

```shell
open-webui confluence-worker --poll-interval 30
```

Check connectivity without changing remote or local state:

```shell
open-webui confluence-import --dry-run --mode full
```

Run one import synchronously:

```shell
open-webui confluence-import --no-dry-run --mode full
```

The connection is disabled by default. Configure and enable it through the
admin-only `/api/v1/integrations/confluence/config` endpoint, then queue a full
or incremental run through `/api/v1/integrations/confluence/sync`.
