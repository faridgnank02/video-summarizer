from src.video_intelligence.agents.chapterizer import ChapterList, Chapterizer
from src.video_intelligence.models.router import Router
from src.video_intelligence.schemas import (
    Chapter, JobOptions, PipelineContext, SourceKind, Transcript, TranscriptOrigin,
    TranscriptSegment, VideoSource,
)
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider

CONFIG = {"tasks": {"chaptering": {"balanced": ["fake/model-x"]}}}


def make(tmp_path):
    fake = FakeProvider("fake")
    router = Router(CONFIG, {"fake": fake}, TraceStore(tmp_path / "t.db"))
    return Chapterizer(router), fake


def ctx_with_transcript(n_segments: int, words_per_segment: int = 5) -> PipelineContext:
    segs = [
        TranscriptSegment(start_s=i * 15.0, end_s=(i + 1) * 15.0,
                          text=" ".join(["word"] * words_per_segment))
        for i in range(n_segments)
    ]
    return PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"),
        options=JobOptions(),
        transcript=Transcript(segments=segs, language="en", origin=TranscriptOrigin.WHISPER),
    )


async def test_single_chunk_produces_chapters(tmp_path):
    chapterizer, fake = make(tmp_path)
    fake.enqueue(ChapterList(chapters=[
        Chapter(start_s=0, end_s=60, title="Intro", synopsis="Opening remarks."),
        Chapter(start_s=60, end_s=150, title="Main", synopsis="The core argument."),
    ]))
    ctx = await chapterizer.run(ctx_with_transcript(10))
    assert [c.title for c in ctx.chapters] == ["Intro", "Main"]
    assert len(fake.calls) == 1
    assert "[0:00]" in fake.calls[0]["prompt"]  # timestamped transcript embedded


async def test_long_transcript_is_chunked_and_results_concatenated(tmp_path):
    chapterizer, fake = make(tmp_path)
    # ~700 segments x ~50 chars ≈ 35k chars -> 2 chunks at MAX_CHARS=24_000
    fake.enqueue(ChapterList(chapters=[Chapter(start_s=0, end_s=1, title="A", synopsis="a")]))
    fake.enqueue(ChapterList(chapters=[Chapter(start_s=1, end_s=2, title="B", synopsis="b")]))
    ctx = await chapterizer.run(ctx_with_transcript(700, words_per_segment=8))
    assert len(fake.calls) == 2
    assert [c.title for c in ctx.chapters] == ["A", "B"]


def test_chapterizer_is_non_essential(tmp_path):
    chapterizer, _ = make(tmp_path)
    assert chapterizer.essential is False
