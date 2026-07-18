"""Test doubles shared across the test suite."""
from __future__ import annotations

from pydantic import BaseModel

from src.video_intelligence.models.providers.base import Provider, Usage


class FakeProvider(Provider):
    def __init__(self, name: str = "fake", available: bool = True):
        self.name = name
        self._available = available
        self._queue: list[BaseModel | Exception] = []
        self.calls: list[dict] = []

    def enqueue(self, item: BaseModel | Exception) -> None:
        self._queue.append(item)

    async def is_available(self) -> bool:
        return self._available

    async def complete(self, model, prompt, schema):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, Usage(tokens_in=100, tokens_out=50)
