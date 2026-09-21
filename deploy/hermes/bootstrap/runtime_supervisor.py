#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BROKER_URL = os.environ['AWG_HERMES_BROKER_URL'].rstrip('/')
TOKEN_FILE = Path('/run/awg-hermes/qwen-token')
RUN_ID_FILE = Path('/run/awg-hermes/qwen-run-id')
SCOPE_ID_FILE = Path('/run/awg-hermes/qwen-scope-id')
ALLOWED_PATH = '/v1/chat/completions'


class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != ALLOWED_PATH:
            self.send_error(404)
            return
        token = TOKEN_FILE.read_text().strip()
        size = int(self.headers.get('content-length', '0'))
        if not token or size <= 0 or size > 4 * 1024 * 1024:
            self.send_error(400)
            return
        request = Request(
            f'{BROKER_URL}/api/v1/integrations/hermes/qwen{ALLOWED_PATH}',
            data=self.rfile.read(size),
            method='POST',
            headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
        )
        request.add_header('X-AWG-Run-Id', RUN_ID_FILE.read_text().strip())
        request.add_header('X-AWG-Scope-Id', SCOPE_ID_FILE.read_text().strip())
        request.add_header('X-AWG-Model', 'awg-qwen')
        try:
            with urlopen(request, timeout=900) as response:
                body = response.read()
                self.send_response(response.status)
                self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except HTTPError as error:
            body = error.read()
            self.send_response(error.code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format, *args):
        return


def main() -> int:
    gateway = subprocess.Popen(['/opt/hermes/.venv/bin/hermes', *sys.argv[1:]])
    server = ThreadingHTTPServer(('127.0.0.1', 8650), ProxyHandler)

    def stop(signum, frame):
        gateway.send_signal(signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    def watch_gateway():
        gateway.wait()
        server.shutdown()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    threading.Thread(target=watch_gateway, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        if gateway.poll() is None:
            gateway.terminate()
        gateway.wait(timeout=15)
    return gateway.returncode


if __name__ == '__main__':
    raise SystemExit(main())
