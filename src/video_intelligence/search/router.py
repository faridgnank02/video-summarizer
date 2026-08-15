"""Config-driven first-available selection over search providers."""
from __future__ import annotations

from .base import NoSearchProvider, SearchProvider, SearchResult


class SearchRouter:
    def __init__(self, config: dict, providers: dict[str, SearchProvider]):
        self._candidates = config.get("search", {}).get("candidates", [])
        self._providers = providers

    async def _pick(self) -> SearchProvider:
        for name in self._candidates:
            provider = self._providers.get(name)
            if provider is not None and await provider.is_available():
                return provider
        raise NoSearchProvider("no search provider available")

    async def search(self, query: str, k: int) -> list[SearchResult]:
        provider = await self._pick()
        return await provider.search(query, k)
