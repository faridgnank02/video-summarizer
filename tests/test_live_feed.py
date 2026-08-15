from src.video_intelligence.live.feed import WindowedTranscriptFeed
from src.video_intelligence.schemas import Transcript, TranscriptOrigin, TranscriptSegment


def make_transcript(n):
    segs = [TranscriptSegment(start_s=i * 5.0, end_s=(i + 1) * 5.0, text=f"seg{i}")
            for i in range(n)]
    return Transcript(segments=segs, language="en", origin=TranscriptOrigin.CAPTIONS)


async def test_feed_yields_all_segments_in_order():
    feed = WindowedTranscriptFeed(make_transcript(3))
    out = [seg async for seg in feed.segments()]
    assert [s.text for s in out] == ["seg0", "seg1", "seg2"]


def test_feed_exposes_language():
    assert WindowedTranscriptFeed(make_transcript(1)).language == "en"
