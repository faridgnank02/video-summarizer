import pytest
from pydantic import BaseModel

from src.video_intelligence.models.providers.base import ProviderError
from src.video_intelligence.models.router import Router, RouterError
from src.video_intelligence.schemas import QualityPreference
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider


class Answer(BaseModel):
    value: int


CONFIG = {
    "tasks": {
        "chaptering": {
            "cheap": ["ollama/llama3.1:8b", "openai/gpt-4o-mini"],
        }
    },
    "pricing": {
        "openai/gpt-4o-mini": {"input_per_mtok": 0.15, "output_per_mtok": 0.60},
    },
}


def make(store, ollama=None, openai=None):
    providers = {"ollama": ollama or FakeProvider("ollama"),
                 "openai": openai or FakeProvider("openai")}
    return Router(CONFIG, providers, store), providers


async def test_uses_first_available_candidate_and_records_span(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    router, providers = make(store)
    providers["ollama"].enqueue(Answer(value=1))

    result = await router.complete(task="chaptering", quality=QualityPreference.CHEAP,
                                   prompt="p", schema=Answer, trace_id="tr", stage="chapterize")
    assert result.value == 1
    span = store.spans("tr")[0]
    assert span.model_used == "ollama/llama3.1:8b"
    assert span.fallback_from is None
    assert span.cost_usd == 0.0  # no pricing entry for local model
    # model passed to the provider keeps its colon
    assert providers["ollama"].calls[0]["model"] == "llama3.1:8b"


async def test_falls_back_when_first_unavailable_and_prices_usage(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    router, providers = make(store, ollama=FakeProvider("ollama", available=False))
    providers["openai"].enqueue(Answer(value=2))

    result = await router.complete(task="chaptering", quality=QualityPreference.CHEAP,
                                   prompt="p", schema=Answer, trace_id="tr", stage="chapterize")
    assert result.value == 2
    span = store.spans("tr")[0]
    assert span.model_used == "openai/gpt-4o-mini"
    assert span.fallback_from == "ollama/llama3.1:8b"
    # FakeProvider reports 100 in / 50 out tokens
    assert span.cost_usd == pytest.approx(100 / 1e6 * 0.15 + 50 / 1e6 * 0.60)


async def test_retries_once_then_falls_back_on_errors(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    router, providers = make(store)
    providers["ollama"].enqueue(ProviderError("flaky"))
    providers["ollama"].enqueue(ProviderError("flaky again"))
    providers["openai"].enqueue(Answer(value=3))

    result = await router.complete(task="chaptering", quality=QualityPreference.CHEAP,
                                   prompt="p", schema=Answer, trace_id="tr", stage="chapterize")
    assert result.value == 3
    assert len(providers["ollama"].calls) == 2  # initial + one retry


async def test_all_candidates_failing_raises_and_records_error_span(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    router, providers = make(store, ollama=FakeProvider("ollama", available=False),
                             openai=FakeProvider("openai", available=False))
    with pytest.raises(RouterError):
        await router.complete(task="chaptering", quality=QualityPreference.CHEAP,
                              prompt="p", schema=Answer, trace_id="tr", stage="chapterize")
    span = store.spans("tr")[0]
    assert span.status == "error"
