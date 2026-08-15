"""Tavily search via httpx (no vendor SDK)."""
from __future__ import annotations

import os

import httpx

from ..base import SearchError, SearchProvider, SearchResult

_ENDPOINT = "https://api.tavily.com/search"


class TavilyProvider(SearchProvider):
    name = "tavily"

    def __init__(self, api_key_env: str = "TAVILY_API_KEY"):
        self._api_key = os.environ.get(api_key_env)
        self._transport: httpx.BaseTransport | None = None  # test seam

    async def is_available(self) -> bool:
        return bool(self._api_key)

    async def search(self, query: str, k: int) -> list[SearchResult]:
        try:
            async with httpx.AsyncClient(timeout=20, transport=self._transport) as client:
                resp = await client.post(_ENDPOINT, json={
                    "api_key": self._api_key, "query": query, "max_results": k,
                })
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise SearchError(f"tavily search failed: {e}") from e
        return [
            SearchResult(title=r.get("title", ""), url=r.get("url", ""),
                         snippet=r.get("content", ""))
            for r in data.get("results", [])
        ]
