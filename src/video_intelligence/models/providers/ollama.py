from __future__ import annotations

import httpx

from .base import Provider, ProviderError, Usage, parse_json_response


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434",
                 transport: httpx.AsyncBaseTransport | None = None):
        self._base_url = base_url
        self._transport = transport

    def _client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    async def is_available(self) -> bool:
        try:
            async with self._client(timeout=2.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def complete(self, model, prompt, schema):
        try:
            async with self._client(timeout=300.0) as client:
                resp = await client.post(f"{self._base_url}/api/chat", json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": "json",
                    "stream": False,
                })
                resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"ollama request failed: {e}") from e
        try:
            data = resp.json()
            content = data["message"]["content"]
            usage = Usage(tokens_in=data.get("prompt_eval_count", 0),
                          tokens_out=data.get("eval_count", 0))
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderError(f"unexpected ollama response shape: {e}") from e
        return parse_json_response(content, schema), usage
