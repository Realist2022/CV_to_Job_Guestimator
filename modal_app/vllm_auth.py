"""ASGI middleware that puts every vLLM route behind the bearer key, not just /v1.

modal_app/inference.py starts vLLM with VLLM_SERVER_DEV_MODE=1, because the
memory-snapshot flow needs the engine's /sleep and /wake_up endpoints. Dev
mode also registers /reset_prefix_cache, /collective_rpc, /is_sleeping and
friends. vLLM's own --api-key only guards paths starting with /v1, and this
server sits on a public Modal URL, so without this anyone could POST /sleep
and take the endpoint down until the next restart.

Loaded into the vLLM process with `--middleware vllm_auth.RequireApiKey`.
/health stays open because it reveals nothing and is handy for checks.
"""

import hmac
import os

_OPEN_PATHS = frozenset({"/health"})


class RequireApiKey:
    def __init__(self, app):
        self.app = app
        self._expected = f"Bearer {os.environ['VLLM_API_KEY']}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") in _OPEN_PATHS:
            await self.app(scope, receive, send)
            return

        supplied = dict(scope.get("headers") or []).get(b"authorization", b"")
        if hmac.compare_digest(supplied, self._expected):
            await self.app(scope, receive, send)
            return

        body = b'{"error":"Unauthorized"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
