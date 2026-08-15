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
    calls = {"downloaded": False, "video_downloaded": False}

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

    def video_downloader(url, workdir: Path):
        calls["video_downloaded"] = True
        out = workdir / "v.mp4"
        out.write_bytes(b"fakevideo")
        return out

    ing = Ingestor(workdir=tmp_path, metadata_fetcher=metadata_fetcher,
                   caption_fetcher=caption_fetcher, audio_downloader=audio_downloader,
                   audio_extractor=audio_extractor, video_downloader=video_downloader)
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


async def test_metadata_fetch_failure_logs_and_continues(tmp_path, caplog):
    def failing_metadata_fetcher(url):
        raise RuntimeError("network down")

    ing, calls = make_ingestor(tmp_path)
    ing._metadata_fetcher = failing_metadata_fetcher
    ctx = ctx_for(VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"))
    with caplog.at_level("WARNING"):
        ctx = await ing.run(ctx)
    assert ctx.transcript is not None  # job still succeeds
    assert any("metadata fetch failed" in rec.message for rec in caplog.records)


async def test_missing_local_file_raises(tmp_path):
    ing, _ = make_ingestor(tmp_path)
    ctx = ctx_for(VideoSource(kind=SourceKind.LOCAL_FILE, path=str(tmp_path / "gone.mp4")))
    with pytest.raises(IngestError):
        await ing.run(ctx)


async def test_visuals_off_does_not_download_video(tmp_path):
    ing, calls = make_ingestor(tmp_path)
    ctx = ctx_for(VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"))
    ctx = await ing.run(ctx)
    assert calls["video_downloaded"] is False
    assert ctx.video_path is None


async def test_visuals_on_downloads_video_even_with_captions(tmp_path):
    ing, calls = make_ingestor(tmp_path)
    ctx = ctx_for(VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"),
                  analyze_visuals=True)
    ctx = await ing.run(ctx)
    assert ctx.transcript is not None          # captions still used
    assert calls["video_downloaded"] is True
    assert ctx.video_path is not None


async def test_visuals_on_local_file_sets_video_path_to_source(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"data")
    ing, calls = make_ingestor(tmp_path)
    ctx = ctx_for(VideoSource(kind=SourceKind.LOCAL_FILE, path=str(src)),
                  analyze_visuals=True)
    ctx = await ing.run(ctx)
    assert ctx.video_path == str(src)


async def test_video_download_failure_is_non_fatal(tmp_path):
    def boom(url, workdir):
        raise RuntimeError("network down")
    ing, calls = make_ingestor(tmp_path)
    ing._video_downloader = boom
    ctx = ctx_for(VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"),
                  analyze_visuals=True)
    ctx = await ing.run(ctx)          # must not raise
    assert ctx.video_path is None
