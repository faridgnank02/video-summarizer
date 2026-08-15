"""Provider interface: one thin async client per model vendor."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Usage(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0


class ProviderError(Exception):
    """Any provider failure: network, API error, unparseable output."""


class Provider(ABC):
    name: str

    @abstractmethod
    async def is_available(self) -> bool: ...

    @abstractmethod
    async def complete(self, model: str, prompt: str, schema: type[T]) -> tuple[T, Usage]: ...


def parse_json_response(text: str, schema: type[T]) -> T:
    """Extract the first JSON object from a model response and validate it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ProviderError(f"no JSON object in response: {text[:200]!r}")
    try:
        return schema.model_validate_json(text[start : end + 1])
    except ValueError as e:
        raise ProviderError(f"schema validation failed: {e}") from e
