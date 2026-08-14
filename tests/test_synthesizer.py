import pytest

from src.video_intelligence.agents.synthesizer import PartialSummary, Synthesizer, SynthesisResult
from src.video_intelligence.models.router import Router
from src.video_intelligence.schemas import (
    Chapter, JobOptions, KeyQuote, PipelineContext, QualityPreference, SourceKind,
    Transcript, TranscriptOrigin, TranscriptSegment, VideoSource,
)
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider

CONFIG = {"tasks": {"synthesis": {
    "balanced": ["fake/big-model"],
    "cheap": ["fake/small-model"],
}}}


def make(tmp_path):
    fake = FakeProvider("fake")
    router = Router(CONFIG, {"fake": fake}, TraceStore(tmp_path / "t.db"))
    return Synthesizer(router), fake


def ctx_with_transcript(n_segments: int, words_per_segment: int = 5) -> PipelineContext:
    segs = [
        TranscriptSegment(start_s=i * 15.0, end_s=(i + 1) * 15.0,
                          text=" ".join(["word"] * words_per_segment))
        for i in range(n_segments)
    ]
    return PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ",
                           title="My Talk"),
        options=JobOptions(),
        transcript=Transcript(segments=segs, language="en", origin=TranscriptOrigin.WHISPER),
        chapters=[Chapter(start_s=0, end_s=60, title="Intro", synopsis="s")],
        degraded_stages=[],
    )


RESULT = SynthesisResult(
    summary="A fine talk.",
    key_quotes=[KeyQuote(timestamp_s=42.0, text="quote")],
    action_items=["do the thing"],
)


async def test_builds_report_from_synthesis(tmp_path):
    synth, fake = make(tmp_path)
    fake.enqueue(RESULT)
    ctx = await synth.run(ctx_with_transcript(10))
    assert ctx.report.summary == "A fine talk."
    assert ctx.report.chapters[0].title == "Intro"
    assert ctx.report.key_quotes[0].timestamp_s == 42.0
    assert ctx.report.trace_id == ctx.trace_id
    assert "My Talk" in fake.calls[0]["prompt"]
    assert len(fake.calls) == 1


async def test_long_transcript_uses_map_reduce(tmp_path):
    synth, fake = make(tmp_path)
    # ~1400 segments ≈ 70k chars -> 2 map calls + 1 final call
    fake.enqueue(PartialSummary(summary="part one"))
    fake.enqueue(PartialSummary(summary="part two"))
    fake.enqueue(RESULT)
    ctx = await synth.run(ctx_with_transcript(1400, words_per_segment=8))
    assert len(fake.calls) == 3
    # map calls go to the cheap tier
    assert fake.calls[0]["model"] == "small-model"
    assert fake.calls[2]["model"] == "big-model"
    assert "part one" in fake.calls[2]["prompt"]
    assert ctx.report.summary == "A fine talk."


async def test_missing_transcript_raises(tmp_path):
    synth, fake = make(tmp_path)
    ctx = ctx_with_transcript(1)
    ctx.transcript = None
    with pytest.raises(ValueError, match="requires a transcript"):
        await synth.run(ctx)
    assert fake.calls == []


async def test_report_carries_degraded_stages(tmp_path):
    synth, fake = make(tmp_path)
    fake.enqueue(RESULT)
    ctx = ctx_with_transcript(10)
    ctx.chapters = None
    ctx.degraded_stages = ["chapterize"]
    ctx = await synth.run(ctx)
    assert ctx.report.chapters == []
    assert ctx.report.degraded_stages == ["chapterize"]
