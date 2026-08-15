"""Task → model routing with availability checks, retry, fallback, tracing."""
from __future__ import annotations

import time
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from ..schemas import QualityPreference, TraceSpan
from ..tracing import TraceStore
from .providers.base import NotSupported, Provider, ProviderError, Usage

T = TypeVar("T", bound=BaseModel)


class RouterError(Exception):
    pass


def load_model_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Router:
    def __init__(self, config: dict, providers: dict[str, Provider], store: TraceStore):
        self._config = config
        self._providers = providers
        self._store = store

    def candidates(self, task: str, quality: QualityPreference) -> list[str]:
        try:
            return self._config["tasks"][task][quality.value]
        except KeyError as e:
            raise RouterError(f"no candidates configured for task={task} quality={quality}") from e

    def _cost(self, candidate: str, usage: Usage) -> float:
        pricing = self._config.get("pricing", {}).get(candidate)
        if not pricing:
            return 0.0
        return (usage.tokens_in / 1e6 * pricing["input_per_mtok"]
                + usage.tokens_out / 1e6 * pricing["output_per_mtok"])

    async def complete(self, *, task: str, quality: QualityPreference, prompt: str,
                       schema: type[T], trace_id: str, stage: str) -> T:
        fallback_from: str | None = None
        errors: list[str] = []
        for candidate in self.candidates(task, quality):
            provider_name, model = candidate.split("/", 1)
            provider = self._providers.get(provider_name)
            if provider is None or not await provider.is_available():
                errors.append(f"{candidate}: unavailable")
                fallback_from = candidate
                continue
            for _attempt in range(2):  # initial call + one retry
                start = time.monotonic()
                try:
                    parsed, usage = await provider.complete(model, prompt, schema)
                except ProviderError as e:
                    errors.append(f"{candidate}: {e}")
                    continue
                self._store.add_span(trace_id, TraceSpan(
                    stage=stage,
                    model_used=candidate,
                    tokens_in=usage.tokens_in,
                    tokens_out=usage.tokens_out,
                    cost_usd=self._cost(candidate, usage),
                    latency_ms=int((time.monotonic() - start) * 1000),
                    status="ok",
                    fallback_from=fallback_from,
                ))
                return parsed
            fallback_from = candidate
        self._store.add_span(trace_id, TraceSpan(stage=stage, model_used="none", status="error"))
        raise RouterError(f"all candidates failed for task={task}: {'; '.join(errors)}")

    async def complete_vision(self, *, task: str, quality: QualityPreference, prompt: str,
                              images: list[bytes], schema: type[T], trace_id: str,
                              stage: str) -> T:
        fallback_from: str | None = None
        errors: list[str] = []
        for candidate in self.candidates(task, quality):
            provider_name, model = candidate.split("/", 1)
            provider = self._providers.get(provider_name)
            if provider is None or not await provider.is_available():
                errors.append(f"{candidate}: unavailable")
                fallback_from = candidate
                continue
            start = time.monotonic()
            try:
                parsed, usage = await provider.complete_vision(model, prompt, images, schema)
            except ProviderError as e:  # includes NotSupported
                errors.append(f"{candidate}: {e}")
                fallback_from = candidate
                continue
            self._store.add_span(trace_id, TraceSpan(
                stage=stage, model_used=candidate, tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out, cost_usd=self._cost(candidate, usage),
                latency_ms=int((time.monotonic() - start) * 1000), status="ok",
                fallback_from=fallback_from))
            return parsed
        self._store.add_span(trace_id, TraceSpan(stage=stage, model_used="none", status="error"))
        raise RouterError(f"all vision candidates failed for task={task}: {'; '.join(errors)}")
