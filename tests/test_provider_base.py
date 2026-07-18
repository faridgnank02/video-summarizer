import pytest
from pydantic import BaseModel

from src.video_intelligence.models.providers.base import ProviderError, Usage, parse_json_response
from tests.fakes import FakeProvider


class Answer(BaseModel):
    value: int


def test_parse_json_response_extracts_embedded_object():
    text = 'Sure! Here is the JSON:\n{"value": 42}\nHope that helps.'
    assert parse_json_response(text, Answer).value == 42


def test_parse_json_response_rejects_missing_json():
    with pytest.raises(ProviderError):
        parse_json_response("no json here", Answer)


def test_parse_json_response_rejects_wrong_shape():
    with pytest.raises(ProviderError):
        parse_json_response('{"other": 1}', Answer)


async def test_fake_provider_returns_queued_items_and_records_calls():
    fake = FakeProvider()
    fake.enqueue(Answer(value=1))
    parsed, usage = await fake.complete("some-model", "prompt text", Answer)
    assert parsed.value == 1
    assert isinstance(usage, Usage)
    assert fake.calls[0]["model"] == "some-model"


async def test_fake_provider_raises_queued_exception():
    fake = FakeProvider()
    fake.enqueue(ProviderError("boom"))
    with pytest.raises(ProviderError):
        await fake.complete("m", "p", Answer)
