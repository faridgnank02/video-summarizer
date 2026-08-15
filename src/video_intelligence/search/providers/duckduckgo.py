"""Keyless DuckDuckGo search via ddgs."""
from __future__ import annotations

import asyncio

from ..base import SearchError, SearchProvider, SearchResult


class DuckDuckGoProvider(SearchProvider):
    name = "duckduckgo"

    async def is_available(self) -> bool:
        return True

    async def search(self, query: str, k: int) -> list[SearchResult]:
        try:
            raw = await asyncio.to_thread(self._search_sync, query, k)
        except Exception as e:  # ddgs raises assorted exceptions on failure
            raise SearchError(f"duckduckgo search failed: {e}") from e
        return [
            SearchResult(title=r.get("title", ""), url=r.get("href", ""),
                         snippet=r.get("body", ""))
            for r in raw
        ]

    def _search_sync(self, query: str, k: int) -> list[dict]:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=k))
