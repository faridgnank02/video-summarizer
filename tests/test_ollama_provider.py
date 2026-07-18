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


async def test_is_available_true_when_tags_responds():
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": []})

    assert await OllamaProvider(transport=make_transport(handler)).is_available() is True


async def test_is_available_false_on_connect_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    assert await OllamaProvider(transport=make_transport(handler)).is_available() is False
