import sys
from types import ModuleType

retrieval = ModuleType('open_webui.integrations.confluence.retrieval')


async def retrieve_confluence_knowledge(*args, **kwargs):
    raise AssertionError('Confluence retrieval is external to Hermes broker unit tests')


retrieval.retrieve_confluence_knowledge = retrieve_confluence_knowledge
sys.modules['open_webui.integrations.confluence.retrieval'] = retrieval

events = ModuleType('open_webui.events')
events.EVENTS = type('Events', (), {'FILE_UPLOADED': 'file.uploaded'})()


async def publish_event(*args, **kwargs):
    return None


events.publish_event = publish_event
sys.modules['open_webui.events'] = events

storage = ModuleType('open_webui.storage.provider')
storage.Storage = type('Storage', (), {})
sys.modules['open_webui.storage.provider'] = storage

loaders = ModuleType('open_webui.retrieval.loaders.main')
loaders.Loader = type('Loader', (), {})
sys.modules['open_webui.retrieval.loaders.main'] = loaders

auth = ModuleType('open_webui.utils.auth')


async def get_verified_user():
    return None


auth.get_verified_user = get_verified_user
sys.modules['open_webui.utils.auth'] = auth
