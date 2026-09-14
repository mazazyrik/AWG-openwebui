from fastapi import APIRouter, Depends, HTTPException, Request, status
from open_webui.integrations.confluence.client import (
    ConfluenceAPIClient,
    ConfluenceClientError,
    ConfluenceMCPClient,
    secret_presence,
)
from open_webui.integrations.confluence.models import (
    ConfluenceConnectionForm,
    ConfluenceConnections,
    ConfluenceSearchForm,
    ConfluenceSyncForm,
)
from open_webui.integrations.confluence.retrieval import retrieve_confluence_knowledge
from open_webui.models.knowledge import KnowledgeForm, Knowledges
from open_webui.utils.auth import get_admin_user

router = APIRouter()


async def _ensure_connection_knowledge(user_id: str, form_data: ConfluenceConnectionForm):
    connection = await ConfluenceConnections.get_default()
    knowledge_id = connection.knowledge_id if connection else None
    knowledge = await Knowledges.get_knowledge_by_id(knowledge_id) if knowledge_id else None

    if not knowledge:
        knowledge = await Knowledges.insert_new_knowledge(
            user_id,
            KnowledgeForm(
                name='Confluence',
                description='Managed Confluence knowledge source',
                access_grants=[],
            ),
        )
        if not knowledge:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to create knowledge')

    meta = {
        **(knowledge.meta or {}),
        'source': 'external',
        'external': {
            'connection_id': 'confluence',
            'provider': 'confluence',
            'managed': True,
            'source': {
                'name': form_data.collection_name,
                'config': {
                    'spaces': form_data.spaces,
                    'citation_fields': ['title', 'url', 'page_id', 'version', 'space', 'hash'],
                },
            },
        },
    }
    await Knowledges.update_knowledge_meta_by_id(knowledge.id, meta)
    return knowledge.id


@router.get('/config')
async def get_config(user=Depends(get_admin_user)):
    connection = await ConfluenceConnections.get_default()
    return {
        'connection': connection.model_dump() if connection else None,
        'secrets': secret_presence(),
    }


@router.put('/config')
async def put_config(form_data: ConfluenceConnectionForm, user=Depends(get_admin_user)):
    knowledge_id = await _ensure_connection_knowledge(user.id, form_data)
    connection = await ConfluenceConnections.upsert_default(form_data, knowledge_id=knowledge_id)
    return {'connection': connection.model_dump(), 'secrets': secret_presence()}


@router.post('/check')
async def check(user=Depends(get_admin_user)):
    connection = await ConfluenceConnections.get_default()
    api = await ConfluenceAPIClient().check()
    mcp = await ConfluenceMCPClient().check()
    ready = bool(
        connection
        and connection.enabled
        and api.get('ok')
        and api.get('enabled')
        and mcp.get('ok')
        and secret_presence().get('CONFLUENCE_QDRANT_URL')
    )
    return {
        'ready': ready,
        'connection': connection.model_dump() if connection else None,
        'api': api,
        'mcp': mcp,
        'secrets': secret_presence(),
        'permissions': {'page_read_check': bool(mcp.get('ok'))},
        'embedding': {
            'model': connection.embedding_model if connection else None,
            'dim': connection.embedding_dim if connection else None,
        },
    }


@router.get('/status')
async def get_status(user=Depends(get_admin_user)):
    connection = await ConfluenceConnections.get_default()
    runs = await ConfluenceConnections.list_runs(connection.id, limit=5) if connection else []
    try:
        remote = await ConfluenceAPIClient().status()
    except ConfluenceClientError:
        remote = None
    return {
        'connection': connection.model_dump() if connection else None,
        'recent_runs': [run.model_dump() for run in runs],
        'remote_run': remote,
    }


@router.get('/runs')
async def runs(user=Depends(get_admin_user)):
    connection = await ConfluenceConnections.get_default()
    if not connection:
        return {'items': []}
    items = await ConfluenceConnections.list_runs(connection.id)
    return {'items': [run.model_dump() for run in items]}


@router.post('/sync')
async def sync(form_data: ConfluenceSyncForm, user=Depends(get_admin_user)):
    connection = await ConfluenceConnections.get_default()
    if not connection:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Confluence connection is not configured')
    run = await ConfluenceConnections.queue_run(connection.id, form_data.mode)
    return run.model_dump()


@router.post('/search')
async def search(request: Request, form_data: ConfluenceSearchForm, user=Depends(get_admin_user)):
    connection = await ConfluenceConnections.get_default()
    if not connection or not connection.knowledge_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Confluence connection is not configured')
    knowledge = await Knowledges.get_knowledge_by_id(connection.knowledge_id)
    if not knowledge:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Confluence knowledge is missing')
    return await retrieve_confluence_knowledge(knowledge, [form_data.query], min(form_data.count, 8), user=user)
