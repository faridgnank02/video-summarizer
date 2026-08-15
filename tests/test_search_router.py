import pytest

from src.video_intelligence.search.base import NoSearchProvider, SearchResult
from src.video_intelligence.search.router import SearchRouter
from tests.fakes import FakeSearch

CONFIG = {"search": {"candidates": ["tavily", "duckduckgo"]}}


async def test_router_picks_first_available():
    tavily = FakeSearch("tavily", available=False)
    ddg = FakeSearch("duckduckgo", available=True)
    ddg.enqueue([SearchResult(title="T", url="https://e.com", snippet="s")])
    router = SearchRouter(CONFIG, {"tavily": tavily, "duckduckgo": ddg})
    results = await router.search("q", k=3)
    assert [r.url for r in results] == ["https://e.com"]
    assert ddg.calls == [{"query": "q", "k": 3}]
    assert tavily.calls == []


async def test_router_raises_when_none_available():
    tavily = FakeSearch("tavily", available=False)
    ddg = FakeSearch("duckduckgo", available=False)
    router = SearchRouter(CONFIG, {"tavily": tavily, "duckduckgo": ddg})
    with pytest.raises(NoSearchProvider):
        await router.search("q", k=3)
