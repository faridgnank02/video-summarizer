import httpx
import pytest
from pydantic import BaseModel

from src.video_intelligence.models.providers.base import ProviderError
from src.video_intelligence.models.providers.ollama import OllamaProvider


class Answer(BaseModel):
    value: int


def make_transport(handler):
    return httpx.MockTransport(handler)


async def test_complete_parses_chat_response():
    def handler(request):
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": '{"value": 7}'},
            "prompt_eval_count": 12,
            "eval_count": 5,
        })

    provider = OllamaProvider(transport=make_transport(handler))
    parsed, usage = await provider.complete("llama3.1:8b", "hi", Answer)
    assert parsed.value == 7
    assert usage.tokens_in == 12
    assert usage.tokens_out == 5


async def test_http_error_becomes_provider_error():
    def handler(request):
        return httpx.Response(500, text="boom")

    provider = OllamaProvider(transport=make_transport(handler))
    with pytest.raises(ProviderError):
        await provider.complete("llama3.1:8b", "hi", Answer)


async def test_malformed_success_body_becomes_provider_error():
    def handler(request):
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = OllamaProvider(transport=make_transport(handler))
    with pytest.raises(ProviderError):
        await provider.complete("llama3.1:8b", "hi", Answer)


async def test_is_available_true_when_tags_responds():
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": []})

    assert await OllamaProvider(transport=make_transport(handler)).is_available() is True


async def test_is_available_false_on_connect_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    assert await OllamaProvider(transport=make_transport(handler)).is_available() is False


def test_base_url_defaults_to_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    provider = OllamaProvider()
    assert provider._base_url == "http://ollama:11434"


def test_base_url_falls_back_to_localhost(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    provider = OllamaProvider()
    assert provider._base_url == "http://localhost:11434"


def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    provider = OllamaProvider(base_url="http://custom:9999")
    assert provider._base_url == "http://custom:9999"
