import pytest

from src.video_intelligence.agents.factchecker import ExtractedClaims, FactChecker, LoopResponse
from src.video_intelligence.models.router import Router
from src.video_intelligence.schemas import (
    AnalysisReport, Claim, ClaimVerdict, QualityPreference,
)
from src.video_intelligence.models.providers.base import ProviderError
from src.video_intelligence.search.base import NoSearchProvider, SearchResult
from src.video_intelligence.search.router import SearchRouter
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider, FakeSearch

CONFIG = {
    "tasks": {"factcheck": {"balanced": ["fake/model-x"]}},
    "search": {"candidates": ["fakesearch"]},
}
BAL = QualityPreference.BALANCED


def make(tmp_path, *, available=True):
    fake_model = FakeProvider("fake")
    router = Router(CONFIG, {"fake": fake_model}, TraceStore(tmp_path / "t.db"))
    fake_search = FakeSearch("fakesearch", available=available)
    search = SearchRouter(CONFIG, {"fakesearch": fake_search})
    checker = FactChecker(router, search, max_claims=8, max_steps=3, results_per_search=5)
    return checker, fake_model, fake_search


def a_report():
    return AnalysisReport(summary="The Eiffel Tower is 330m tall.",
                          language="en", trace_id="tr1")


async def test_extract_returns_claims_capped(tmp_path):
    checker, model, _ = make(tmp_path)
    checker._max_claims = 2
    model.enqueue(ExtractedClaims(claims=[
        Claim(text="c1"), Claim(text="c2"), Claim(text="c3"),
    ]))
    claims = await checker.extract_claims(a_report(), None, BAL, "tr1")
    assert [c.text for c in claims] == ["c1", "c2"]


async def test_verify_reaches_supported_verdict(tmp_path):
    checker, model, search = make(tmp_path)
    search.enqueue([SearchResult(title="T", url="https://e.com", snippet="330 metres")])
    model.enqueue(LoopResponse(action="verdict", verdict=ClaimVerdict.SUPPORTED,
                               confidence=0.9, rationale="matches source",
                               cited_urls=["https://e.com"]))
    fc = await checker.verify_claim(Claim(text="Eiffel Tower is 330m"), BAL, "tr1")
    assert fc.verdict == "supported"
    assert fc.search_steps == 1
    assert [e.url for e in fc.evidence] == ["https://e.com"]


async def test_verify_refine_triggers_second_search(tmp_path):
    checker, model, search = make(tmp_path)
    search.enqueue([SearchResult(title="A", url="https://a.com", snippet="vague")])
    search.enqueue([SearchResult(title="B", url="https://b.com", snippet="precise")])
    model.enqueue(LoopResponse(action="search", query="Eiffel Tower height metres"))
    model.enqueue(LoopResponse(action="verdict", verdict=ClaimVerdict.REFUTED,
                               rationale="source says 300m"))
    fc = await checker.verify_claim(Claim(text="Eiffel Tower is 330m"), BAL, "tr1")
    assert fc.verdict == "refuted"
    assert fc.search_steps == 2
    assert search.calls[1]["query"] == "Eiffel Tower height metres"


async def test_verify_budget_exhausted_is_unverified(tmp_path):
    checker, model, search = make(tmp_path)
    for _ in range(3):
        search.enqueue([SearchResult(title="T", url="https://e.com", snippet="s")])
        model.enqueue(LoopResponse(action="search", query="again"))
    fc = await checker.verify_claim(Claim(text="claim"), BAL, "tr1")
    assert fc.verdict == "unverified"
    assert fc.search_steps == 3


async def test_verify_no_provider_bubbles_up(tmp_path):
    checker, _, _ = make(tmp_path, available=False)
    with pytest.raises(NoSearchProvider):
        await checker.verify_claim(Claim(text="claim"), BAL, "tr1")


async def test_extract_returns_empty_on_router_error(tmp_path):
    checker, model, _ = make(tmp_path)
    model.enqueue(ProviderError("boom"))
    model.enqueue(ProviderError("boom"))
    claims = await checker.extract_claims(a_report(), None, BAL, "tr1")
    assert claims == []


async def test_verify_router_error_mid_loop_is_unverified(tmp_path):
    checker, model, search = make(tmp_path)
    search.enqueue([SearchResult(title="T", url="https://e.com", snippet="s")])
    model.enqueue(ProviderError("boom"))
    model.enqueue(ProviderError("boom"))
    fc = await checker.verify_claim(Claim(text="c"), BAL, "tr1")
    assert fc.verdict == "unverified"
    assert [e.url for e in fc.evidence] == ["https://e.com"]


async def test_check_wraps_bare_strings(tmp_path):
    checker, model, search = make(tmp_path)
    search.enqueue([SearchResult(title="T", url="https://e.com", snippet="s")])
    model.enqueue(LoopResponse(action="verdict", verdict=ClaimVerdict.MISLEADING,
                               rationale="missing context"))
    out = await checker.check(["some claim"], BAL, "tr1")
    assert len(out) == 1 and out[0].verdict == "misleading"
