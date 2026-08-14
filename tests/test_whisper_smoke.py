from pathlib import Path

import pytest

from src.video_intelligence.agents.transcriber import Transcriber
from src.video_intelligence.schemas import (
    JobOptions, PipelineContext, SourceKind, TranscriptOrigin, VideoSource,
)

FIXTURE = Path(__file__).parent / "fixtures" / "spoken_30s.wav"


@pytest.mark.slow
@pytest.mark.skipif(not FIXTURE.exists(), reason="run scripts/make_fixture.sh first")
async def test_real_whisper_transcribes_fixture():
    ctx = PipelineContext(
        source=VideoSource(kind=SourceKind.LOCAL_FILE, path=str(FIXTURE)),
        options=JobOptions(),
        audio_path=str(FIXTURE),
    )
    ctx = await Transcriber(model_name="base").run(ctx)
    assert ctx.transcript.origin == TranscriptOrigin.WHISPER
    text = ctx.transcript.full_text.lower()
    assert "pipeline" in text
    assert "whisper" in text
    assert ctx.transcript.segments[0].start_s < 2.0
