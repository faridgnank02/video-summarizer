import pytest

from src.video_intelligence.agents.base import Agent
from src.video_intelligence.agents.chapterizer import ChapterList, Chapterizer
from src.video_intelligence.agents.ingestor import Ingestor
from src.video_intelligence.agents.synthesizer import Synthesizer, SynthesisResult
from src.video_intelligence.agents.transcriber import Transcriber
from src.video_intelligence.models.providers.base import ProviderError
from src.video_intelligence.models.router import Router
from src.video_intelligence.pipeline import Pipeline, PipelineError
from src.video_intelligence.schemas import Chapter, JobOptions, SourceKind, VideoSource
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider

RAW_CAPTIONS = [
    {"text": "hello", "start": 0.0, "duration": 2.0},
    {"text": "world", "start": 2.0, "duration": 2.0},
]

CONFIG = {"tasks": {
    "chaptering": {"balanced": ["fake/small"]},
    "synthesis": {"balanced": ["fake/big"], "cheap": ["fake/small"]},
}}


def build_test_pipeline(tmp_path, fake: FakeProvider, on_event=None) -> Pipeline:
    store = TraceStore(tmp_path / "t.db")
    router = Router(CONFIG, {"fake": fake}, store)
    ingestor = Ingestor(
        workdir=tmp_path,
        metadata_fetcher=lambda url: {"title": "T", "duration_s": 4.0, "channel": "C"},
        caption_fetcher=lambda vid, lang: RAW_CAPTIONS,
        audio_downloader=lambda url, wd: (_ for _ in ()).throw(AssertionError("no download")),
        audio_extractor=lambda p, wd: (_ for _ in ()).throw(AssertionError("no extract")),
    )
    transcriber = Transcriber(model_factory=lambda name: (_ for _ in ()).throw(
        AssertionError("whisper must not load")))
    return Pipeline(
        [ingestor, transcriber, Chapterizer(router), Synthesizer(router)],
        on_event=on_event,
    ), store


def youtube_source() -> VideoSource:
    return VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ")


async def test_full_pipeline_produces_report_and_events(tmp_path):
    fake = FakeProvider("fake")
    fake.enqueue(ChapterList(chapters=[Chapter(start_s=0, end_s=4, title="All", synopsis="s")]))
    fake.enqueue(SynthesisResult(summary="Nice video."))
    events = []

    async def on_event(ev):
        events.append((ev.stage, ev.type))

    pipeline, store = build_test_pipeline(tmp_path, fake, on_event=on_event)
    report = await pipeline.run(youtube_source(), JobOptions())

    assert report.summary == "Nice video."
    assert report.chapters[0].title == "All"
    assert report.degraded_stages == []
    # every stage traced or evented
    assert ("ingest", "started") in events
    assert ("synthesize", "completed") in events
    # spans exist for the two model calls
    stages = [s.stage for s in store.spans(report.trace_id)]
    assert stages == ["chapterize", "synthesize"]


async def test_non_essential_failure_degrades_but_completes(tmp_path):
    fake = FakeProvider("fake")
    # chapterizer: initial + retry both fail -> RouterError -> degraded
    fake.enqueue(ProviderError("boom"))
    fake.enqueue(ProviderError("boom"))
    fake.enqueue(SynthesisResult(summary="Still fine."))
    pipeline, _ = build_test_pipeline(tmp_path, fake)
    report = await pipeline.run(youtube_source(), JobOptions())
    assert report.summary == "Still fine."
    assert report.chapters == []
    assert report.degraded_stages == ["chapterize"]


async def test_essential_failure_raises_pipeline_error(tmp_path):
    fake = FakeProvider("fake")
    fake.enqueue(ChapterList(chapters=[]))
    fake.enqueue(ProviderError("boom"))
    fake.enqueue(ProviderError("boom"))
    pipeline, _ = build_test_pipeline(tmp_path, fake)
    with pytest.raises(PipelineError) as exc:
        await pipeline.run(youtube_source(), JobOptions())
    assert exc.value.stage == "synthesize"
