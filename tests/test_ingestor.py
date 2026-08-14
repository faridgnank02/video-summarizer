from pathlib import Path

import pytest

from src.video_intelligence.agents.ingestor import Ingestor, IngestError, extract_video_id
from src.video_intelligence.schemas import (
    JobOptions, PipelineContext, SourceKind, TranscriptOrigin, VideoSource,
)

RAW_CAPTIONS = [
    {"text": "hello", "start": 0.0, "duration": 2.0},
    {"text": "world", "start": 2.0, "duration": 2.0},
]


def ctx_for(source: VideoSource, **opts) -> PipelineContext:
    return PipelineContext(source=source, options=JobOptions(**opts))


def make_ingestor(tmp_path, captions=RAW_CAPTIONS, downloads=None):
    calls = {"downloaded": False}

    def metadata_fetcher(url):
        return {"title": "T", "duration_s": 120.0, "channel": "C"}

    def caption_fetcher(video_id, language):
        return captions

    def audio_downloader(url, workdir: Path):
        calls["downloaded"] = True
        out = workdir / "a.m4a"
        out.write_bytes(b"fake")
        return out

    def audio_extractor(path, workdir: Path):
        out = workdir / "a.wav"
        out.write_bytes(b"fake")
        return out

    ing = Ingestor(workdir=tmp_path, metadata_fetcher=metadata_fetcher,
                   caption_fetcher=caption_fetcher, audio_downloader=audio_downloader,
                   audio_extractor=audio_extractor)
    return ing, calls


def test_extract_video_id_handles_common_forms():
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    with pytest.raises(IngestError):
        extract_video_id("https://example.com/nope")


async def test_youtube_with_captions_skips_download(tmp_path):
    ing, calls = make_ingestor(tmp_path)
    ctx = ctx_for(VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"))
    ctx = await ing.run(ctx)
    assert ctx.transcript is not None
    assert ctx.transcript.origin == TranscriptOrigin.CAPTIONS
    assert ctx.transcript.segments[1].text == "world"
    assert ctx.transcript.segments[1].end_s == 4.0
    assert ctx.audio_path is None
    assert calls["downloaded"] is False
    assert ctx.source.title == "T"  # metadata resolved


async def test_youtube_without_captions_downloads_audio(tmp_path):
    ing, calls = make_ingestor(tmp_path, captions=None)
    ctx = ctx_for(VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"))
    ctx = await ing.run(ctx)
    assert ctx.transcript is None
    assert ctx.audio_path is not None and calls["downloaded"] is True


async def test_force_whisper_ignores_captions(tmp_path):
    ing, calls = make_ingestor(tmp_path)
    ctx = ctx_for(VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"),
                  force_whisper=True)
    ctx = await ing.run(ctx)
    assert ctx.transcript is None and calls["downloaded"] is True


async def test_local_file_extracts_audio(tmp_path):
    ing, _ = make_ingestor(tmp_path)
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"fake video")
    ctx = ctx_for(VideoSource(kind=SourceKind.LOCAL_FILE, path=str(video)))
    ctx = await ing.run(ctx)
    assert ctx.audio_path.endswith(".wav")


async def test_missing_local_file_raises(tmp_path):
    ing, _ = make_ingestor(tmp_path)
    ctx = ctx_for(VideoSource(kind=SourceKind.LOCAL_FILE, path=str(tmp_path / "gone.mp4")))
    with pytest.raises(IngestError):
        await ing.run(ctx)
