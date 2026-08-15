import pytest

from src.video_intelligence.live.feed import WindowedTranscriptFeed
from src.video_intelligence.live.summarizer import RollingResult, RollingSummarizer
from src.video_intelligence.models.providers.base import ProviderError
from src.video_intelligence.models.router import Router
from src.video_intelligence.schemas import (
    QualityPreference, StageEvent, Transcript, TranscriptOrigin, TranscriptSegment,
)
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider

CONFIG = {"tasks": {"rolling": {"balanced": ["fake/m"]}}}
BAL = QualityPreference.BALANCED


def make(tmp_path):
    fake = FakeProvider("fake")
    router = Router(CONFIG, {"fake": fake}, TraceStore(tmp_path / "t.db"))
    events: list[StageEvent] = []

    async def on_event(ev):
        events.append(ev)

    summ = RollingSummarizer(router, window_s=10, quality=BAL, on_event=on_event)
    return summ, fake, events


def feed_of(spans):
    # spans: list of (start_s, end_s)
    segs = [TranscriptSegment(start_s=s, end_s=e, text=f"t{i}")
            for i, (s, e) in enumerate(spans)]
    return WindowedTranscriptFeed(
        Transcript(segments=segs, language="en", origin=TranscriptOrigin.CAPTIONS))


async def test_two_windows_emit_ordered_summary_events(tmp_path):
    summ, fake, events = make(tmp_path)
    # window 1 closes at seg spanning >=10s; window 2 is the flushed tail.
    # per window: delta call (+ fold call only when prior digest non-empty)
    fake.enqueue(RollingResult(summary="delta1"))          # window0 delta (digest empty -> adopts delta1)
    fake.enqueue(RollingResult(summary="delta2"))          # window1 delta
    fake.enqueue(RollingResult(summary="digest2"))         # window1 fold
    fake.enqueue(RollingResult(summary="final"))           # consolidate
    report = await summ.run(feed_of([(0, 5), (5, 12), (12, 15)]), "tr1")
    summaries = [e for e in events if e.type == "summary"]
    assert [e.data["window_index"] for e in summaries] == [0, 1]
    assert summaries[0].data["running_summary"] == "delta1"
    assert summaries[1].data["running_summary"] == "digest2"
    assert summaries[1].data["window_start_s"] == 12.0
    assert report.summary == "final"


async def test_window_bounds_reflect_segments(tmp_path):
    summ, fake, events = make(tmp_path)
    fake.enqueue(RollingResult(summary="d0"))
    fake.enqueue(RollingResult(summary="final"))
    await summ.run(feed_of([(0, 4), (4, 11)]), "tr1")
    ev = [e for e in events if e.type == "summary"][0]
    assert ev.data["window_start_s"] == 0.0 and ev.data["window_end_s"] == 11.0


async def test_router_error_emits_gap_and_continues(tmp_path):
    summ, fake, events = make(tmp_path)
    fake.enqueue(ProviderError("boom"))    # window0 delta fails (initial)
    fake.enqueue(ProviderError("boom"))    # window0 delta retry -> RouterError
    report = await summ.run(feed_of([(0, 11)]), "tr1")
    summaries = [e for e in events if e.type == "summary"]
    assert len(summaries) == 1
    assert summaries[0].data["delta"] == "(summary unavailable for this window)"
    assert summaries[0].data["running_summary"] == ""   # digest unchanged (was empty)
    # digest stayed empty, so consolidation is skipped and summary is empty
    assert report.summary == ""
    assert len(fake.calls) == 2   # only the two failed delta attempts


async def test_empty_transcript_no_events_empty_summary(tmp_path):
    summ, fake, events = make(tmp_path)
    report = await summ.run(feed_of([]), "tr1")
    assert [e for e in events if e.type == "summary"] == []
    assert report.summary == ""
    assert fake.calls == []
