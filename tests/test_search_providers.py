import httpx
import pytest

from src.video_intelligence.search.base import SearchError
from src.video_intelligence.search.providers.tavily import TavilyProvider


async def test_tavily_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = TavilyProvider()
    assert await provider.is_available() is False


async def test_tavily_available_with_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    assert await TavilyProvider().is_available() is True


async def test_tavily_parses_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [
            {"title": "A", "url": "https://a.com", "content": "alpha"},
        ]})

    provider = TavilyProvider()
    provider._transport = httpx.MockTransport(handler)  # test seam
    results = await provider.search("q", k=3)
    assert results[0].url == "https://a.com"
    assert results[0].snippet == "alpha"


async def test_tavily_wraps_http_errors(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    provider = TavilyProvider()
    provider._transport = httpx.MockTransport(handler)
    with pytest.raises(SearchError):
        await provider.search("q", k=3)


@pytest.mark.slow
async def test_duckduckgo_real_search():
    from src.video_intelligence.search.providers.duckduckgo import DuckDuckGoProvider
    results = await DuckDuckGoProvider().search("python programming language", k=3)
    assert results and all(r.url for r in results)
