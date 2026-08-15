import pytest
from pydantic import BaseModel

from src.video_intelligence.models.providers.anthropic import AnthropicProvider
from src.video_intelligence.models.providers.base import ProviderError
from src.video_intelligence.models.providers.openai import OpenAIProvider


class Answer(BaseModel):
    value: int


# --- minimal stubs shaped like each vendor SDK's response objects ---

class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class StubOpenAIClient:
    def __init__(self, content='{"value": 3}', error: Exception | None = None):
        outer = self
        class _Completions:
            async def create(self, **kwargs):
                outer.kwargs = kwargs
                if error:
                    raise error
                return _Obj(
                    choices=[_Obj(message=_Obj(content=content))],
                    usage=_Obj(prompt_tokens=10, completion_tokens=4),
                )
        self.chat = _Obj(completions=_Completions())


class StubAnthropicClient:
    def __init__(self, content='{"value": 9}', error: Exception | None = None):
        outer = self
        class _Messages:
            async def create(self, **kwargs):
                outer.kwargs = kwargs
                if error:
                    raise error
                return _Obj(
                    content=[_Obj(text=content)],
                    usage=_Obj(input_tokens=20, output_tokens=6),
                )
        self.messages = _Messages()


async def test_openai_complete_parses_and_reports_usage():
    stub = StubOpenAIClient()
    parsed, usage = await OpenAIProvider(client=stub).complete("gpt-4o-mini", "p", Answer)
    assert parsed.value == 3
    assert (usage.tokens_in, usage.tokens_out) == (10, 4)
    assert stub.kwargs["model"] == "gpt-4o-mini"


async def test_openai_error_becomes_provider_error():
    stub = StubOpenAIClient(error=RuntimeError("api down"))
    with pytest.raises(ProviderError):
        await OpenAIProvider(client=stub).complete("gpt-4o-mini", "p", Answer)


async def test_anthropic_complete_parses_and_reports_usage():
    stub = StubAnthropicClient()
    parsed, usage = await AnthropicProvider(client=stub).complete("claude-sonnet", "p", Answer)
    assert parsed.value == 9
    assert (usage.tokens_in, usage.tokens_out) == (20, 6)


async def test_openai_malformed_response_becomes_provider_error():
    class _MalformedOpenAIClient:
        def __init__(self):
            class _Completions:
                async def create(self, **kwargs):
                    return _Obj(choices=[], usage=None)
            self.chat = _Obj(completions=_Completions())

    with pytest.raises(ProviderError):
        await OpenAIProvider(client=_MalformedOpenAIClient()).complete("gpt-4o-mini", "p", Answer)


async def test_anthropic_malformed_response_becomes_provider_error():
    class _MalformedAnthropicClient:
        def __init__(self):
            class _Messages:
                async def create(self, **kwargs):
                    return _Obj(content=[], usage=None)
            self.messages = _Messages()

    with pytest.raises(ProviderError):
        await AnthropicProvider(client=_MalformedAnthropicClient()).complete("claude-sonnet", "p", Answer)


async def test_availability_follows_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert await OpenAIProvider().is_available() is False
    assert await AnthropicProvider().is_available() is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-y")
    assert await OpenAIProvider().is_available() is True
    assert await AnthropicProvider().is_available() is True
