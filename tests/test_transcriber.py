from types import SimpleNamespace

import pytest

from src.video_intelligence.agents.transcriber import Transcriber, TranscribeError
from src.video_intelligence.schemas import (
    JobOptions, PipelineContext, SourceKind, Transcript, TranscriptOrigin,
    TranscriptSegment, VideoSource,
)


class FakeWhisperModel:
    def transcribe(self, audio_path):
        segments = [
            SimpleNamespace(start=0.0, end=3.0, text=" Hello there. "),
            SimpleNamespace(start=3.0, end=6.0, text=" General Kenobi. "),
        ]
        return iter(segments), SimpleNamespace(language="en")


def base_ctx(**kwargs) -> PipelineContext:
    return PipelineContext(
        source=VideoSource(kind=SourceKind.LOCAL_FILE, path="x.mp4"),
        options=JobOptions(), **kwargs,
    )


async def test_transcribes_audio_with_whisper():
    t = Transcriber(model_factory=lambda name: FakeWhisperModel())
    ctx = base_ctx(audio_path="/tmp/a.wav")
    ctx = await t.run(ctx)
    assert ctx.transcript.origin == TranscriptOrigin.WHISPER
    assert ctx.transcript.language == "en"
    assert ctx.transcript.segments[0].text == "Hello there."
    assert ctx.transcript.segments[1].end_s == 6.0


async def test_skips_when_captions_already_present():
    def exploding_factory(name):
        raise AssertionError("whisper must not be loaded when captions exist")
    t = Transcriber(model_factory=exploding_factory)
    existing = Transcript(
        segments=[TranscriptSegment(start_s=0, end_s=1, text="cap")],
        language="en", origin=TranscriptOrigin.CAPTIONS,
    )
    ctx = base_ctx(transcript=existing)
    ctx = await t.run(ctx)
    assert ctx.transcript is existing


async def test_no_audio_and_no_transcript_raises():
    t = Transcriber(model_factory=lambda name: FakeWhisperModel())
    with pytest.raises(TranscribeError):
        await t.run(base_ctx())
