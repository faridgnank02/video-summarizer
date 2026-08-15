import pytest

from src.video_intelligence.agents.base import Agent
from src.video_intelligence.agents.factchecker import (
    ExtractedClaims, FactChecker, FactCheckerAgent, LoopResponse,
)
from src.video_intelligence.models.router import Router
from src.video_intelligence.schemas import (
    AnalysisReport, Claim, ClaimVerdict, JobOptions, PipelineContext, SourceKind,
    VideoSource,
)
from src.video_intelligence.search.base import SearchResult
from src.video_intelligence.search.router import SearchRouter
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider, FakeSearch

CONFIG = {
    "tasks": {"factcheck": {"balanced": ["fake/model-x"]}},
    "search": {"candidates": ["fakesearch"]},
}


def make_agent(tmp_path, *, search_available=True):
    model = FakeProvider("fake")
    router = Router(CONFIG, {"fake": model}, TraceStore(tmp_path / "t.db"))
    search = SearchRouter(CONFIG, {"fakesearch": FakeSearch("fakesearch", available=search_available)})
    checker = FactChecker(router, search)
    return FactCheckerAgent(checker), model, search


def ctx_with_report(fact_check: bool):
    report = AnalysisReport(summary="A claim.", language="en", trace_id="tr1")
    return PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/x"),
        options=JobOptions(fact_check=fact_check),
        report=report, trace_id="tr1",
    )


def test_agent_is_non_essential(tmp_path):
    agent, _, _ = make_agent(tmp_path)
    assert agent.essential is False and agent.name == "fact_check"


async def test_agent_noop_when_flag_off(tmp_path):
    agent, model, search = make_agent(tmp_path)
    ctx = await agent.run(ctx_with_report(fact_check=False))
    assert ctx.report.fact_checks == []
    assert model.calls == []


async def test_agent_populates_fact_checks_when_on(tmp_path):
    agent, model, search = make_agent(tmp_path)
    fake_search = next(iter(search._providers.values()))
    model.enqueue(ExtractedClaims(claims=[Claim(text="A claim.")]))
    fake_search.enqueue([SearchResult(title="T", url="https://e.com", snippet="s")])
    model.enqueue(LoopResponse(action="verdict", verdict=ClaimVerdict.SUPPORTED,
                               rationale="ok"))
    ctx = await agent.run(ctx_with_report(fact_check=True))
    assert [fc.verdict for fc in ctx.report.fact_checks] == ["supported"]


async def test_agent_raises_when_search_unavailable(tmp_path):
    from src.video_intelligence.search.base import NoSearchProvider
    agent, model, _ = make_agent(tmp_path, search_available=False)
    model.enqueue(ExtractedClaims(claims=[Claim(text="A claim.")]))
    with pytest.raises(NoSearchProvider):
        await agent.run(ctx_with_report(fact_check=True))


async def test_pipeline_degrades_when_search_unavailable(tmp_path):
    from src.video_intelligence.pipeline import Pipeline

    class ReportStub(Agent):
        name = "synthesize"
        essential = True

        async def run(self, ctx):
            ctx.report = AnalysisReport(summary="s", language="en", trace_id=ctx.trace_id)
            return ctx

    agent, model, _ = make_agent(tmp_path, search_available=False)
    model.enqueue(ExtractedClaims(claims=[Claim(text="A claim.")]))
    pipeline = Pipeline([ReportStub(), agent])
    report = await pipeline.run(
        VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/x"),
        JobOptions(fact_check=True),
    )
    assert "fact_check" in report.degraded_stages
