import pytest

from src.video_intelligence.agents.rolling import RollingSummarizerAgent
from src.video_intelligence.models.router import Router
from src.video_intelligence.schemas import (
    JobOptions, PipelineContext, SourceKind, Transcript, TranscriptOrigin,
    TranscriptSegment, VideoSource,
)
from src.video_intelligence.live.summarizer import RollingResult
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider

CONFIG = {"tasks": {"rolling": {"balanced": ["fake/m"]}}}


def ctx_with_transcript():
    segs = [TranscriptSegment(start_s=i * 5.0, end_s=(i + 1) * 5.0, text=f"s{i}")
            for i in range(3)]
    return PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/x"),
        options=JobOptions(live=True),
        transcript=Transcript(segments=segs, language="en", origin=TranscriptOrigin.CAPTIONS),
        trace_id="tr1")


def test_agent_is_essential():
    router = Router(CONFIG, {}, None)
    agent = RollingSummarizerAgent(router)
    assert agent.essential is True and agent.name == "rolling_summarize"


async def test_agent_sets_report_and_emits(tmp_path):
    fake = FakeProvider("fake")
    router = Router(CONFIG, {"fake": fake}, TraceStore(tmp_path / "t.db"))
    events = []

    async def on_event(ev):
        events.append(ev)

    fake.enqueue(RollingResult(summary="d0"))     # single window delta
    fake.enqueue(RollingResult(summary="final"))  # consolidate
    agent = RollingSummarizerAgent(router, on_event=on_event, window_s=15)
    ctx = await agent.run(ctx_with_transcript())
    assert ctx.report.summary == "final"
    assert any(e.type == "summary" for e in events)


async def test_agent_requires_transcript(tmp_path):
    router = Router(CONFIG, {"fake": FakeProvider()}, TraceStore(tmp_path / "t.db"))
    ctx = ctx_with_transcript()
    ctx.transcript = None
    with pytest.raises(ValueError, match="requires a transcript"):
        await RollingSummarizerAgent(router).run(ctx)
