from __future__ import annotations

import json
import base64
import mimetypes
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mcp.server.fastmcp import FastMCP

BROKER_URL = os.environ['AWG_HERMES_BROKER_URL'].rstrip('/')
TOKEN_FILE = Path(os.environ.get('AWG_HERMES_PRINCIPAL_FILE', '/run/awg-hermes/principal-token'))
WORKSPACE = Path('/workspace').resolve()
ALLOWED_ARTIFACTS = {'.pdf', '.docx', '.xlsx', '.pptx'}
MAX_ARTIFACT_BYTES = 50 * 1024 * 1024

mcp = FastMCP('awg')


def _request(path: str, *, method: str = 'GET', payload: dict | None = None, body: bytes | None = None, headers=None):
    request_headers = {'Authorization': f'Bearer {TOKEN_FILE.read_text().strip()}', **(headers or {})}
    if payload is not None:
        body = json.dumps(payload).encode()
        request_headers['Content-Type'] = 'application/json'
    request = Request(f'{BROKER_URL}{path}', data=body, method=method, headers=request_headers)
    try:
        with urlopen(request, timeout=120) as response:
            content = response.read()
            content_type = response.headers.get_content_type()
            return json.loads(content) if content_type == 'application/json' else content
    except HTTPError as error:
        raise RuntimeError(f'AWG broker returned HTTP {error.code}') from error


@mcp.tool()
def search_confluence(query: str, limit: int = 6) -> dict:
    """Search the read-only AWG Confluence index."""
    return _request(
        '/api/v1/integrations/hermes/tools/search_confluence', method='POST', payload={'query': query, 'limit': limit}
    )


@mcp.tool()
def read_attachment(file_id: str) -> str:
    """Download an authorized OpenWebUI attachment into this workspace."""
    metadata = _request(
        '/api/v1/integrations/hermes/tools/read_attachment', method='POST', payload={'file_id': file_id}
    )
    destination = (WORKSPACE / 'attachments' / Path(metadata['filename']).name).resolve()
    if not destination.is_relative_to(WORKSPACE):
        raise ValueError('Invalid attachment path')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_request(metadata['download_url']))
    return str(destination)


@mcp.tool()
def publish_artifact(path: str) -> dict:
    """Publish a generated PDF, DOCX, XLSX, or PPTX to OpenWebUI."""
    source = Path(path).resolve()
    if not source.is_relative_to(WORKSPACE) or source.suffix.lower() not in ALLOWED_ARTIFACTS:
        raise ValueError('Artifact must be an allowed file inside /workspace')
    if not source.is_file() or source.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError('Artifact is missing or too large')
    return _request(
        '/api/v1/integrations/hermes/artifacts',
        method='POST',
        body=source.read_bytes(),
        headers={
            'Content-Type': mimetypes.guess_type(source.name)[0] or 'application/octet-stream',
            'X-AWG-Filename-B64': base64.urlsafe_b64encode(source.name.encode('utf-8')).decode('ascii'),
        },
    )


@mcp.tool()
def stage_plugin(name: str, files: dict[str, str]) -> dict:
    """Validate and activate a plugin through the protected AWG supervisor."""
    return _request(
        '/api/v1/integrations/hermes/tools/stage_plugin',
        method='POST',
        payload={'name': name, 'files': files},
    )


if __name__ == '__main__':
    mcp.run()
