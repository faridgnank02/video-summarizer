"""Search provider interface: one thin async client per search backend."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    content: str | None = None


class SearchError(Exception):
    """A search backend failed for a specific query."""


class NoSearchProvider(SearchError):
    """No configured search backend is available at all."""


class SearchProvider(ABC):
    name: str

    @abstractmethod
    async def is_available(self) -> bool: ...

    @abstractmethod
    async def search(self, query: str, k: int) -> list[SearchResult]: ...
