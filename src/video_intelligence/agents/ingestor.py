"""Resolve a video source into either captions or an audio file to transcribe."""
from __future__ import annotations

import asyncio
import re
import subprocess
import uuid
from pathlib import Path

from ..schemas import (
    PipelineContext, SourceKind, Transcript, TranscriptOrigin, TranscriptSegment,
)
from .base import Agent


class IngestError(Exception):
    pass


_YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})")


def extract_video_id(url: str) -> str:
    m = _YOUTUBE_ID_RE.search(url)
    if not m:
        raise IngestError(f"could not extract YouTube video id from {url!r}")
    return m.group(1)


def default_metadata_fetcher(url: str) -> dict:
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return {"title": info.get("title"), "duration_s": info.get("duration"),
            "channel": info.get("channel")}


def default_caption_fetcher(video_id: str, language: str) -> list[dict] | None:
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        return YouTubeTranscriptApi().fetch(video_id, languages=[language]).to_raw_data()
    except Exception:
        return None


def default_audio_downloader(url: str, workdir: Path) -> Path:
    import yt_dlp
    stem = uuid.uuid4().hex
    with yt_dlp.YoutubeDL({
        "quiet": True,
        "format": "bestaudio/best",
        "outtmpl": str(workdir / f"{stem}.%(ext)s"),
    }) as ydl:
        ydl.download([url])
    files = list(workdir.glob(f"{stem}.*"))
    if not files:
        raise IngestError(f"yt-dlp produced no audio file for {url!r}")
    return files[0]


def default_audio_extractor(path: str, workdir: Path) -> Path:
    out = workdir / f"{uuid.uuid4().hex}.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", str(out)],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise IngestError(f"ffmpeg audio extraction failed: {e}") from e
    return out


class Ingestor(Agent):
    name = "ingest"
    essential = True

    def __init__(self, workdir: str | Path = "data/work",
                 metadata_fetcher=default_metadata_fetcher,
                 caption_fetcher=default_caption_fetcher,
                 audio_downloader=default_audio_downloader,
                 audio_extractor=default_audio_extractor):
        self._workdir = Path(workdir)
        self._metadata_fetcher = metadata_fetcher
        self._caption_fetcher = caption_fetcher
        self._audio_downloader = audio_downloader
        self._audio_extractor = audio_extractor

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        self._workdir.mkdir(parents=True, exist_ok=True)
        if ctx.source.kind == SourceKind.YOUTUBE:
            await self._ingest_youtube(ctx)
        else:
            await self._ingest_local(ctx)
        return ctx

    async def _ingest_youtube(self, ctx: PipelineContext) -> None:
        try:
            meta = await asyncio.to_thread(self._metadata_fetcher, ctx.source.url)
            ctx.source = ctx.source.model_copy(update=meta)
        except Exception:
            pass  # metadata is nice-to-have; never fail the job over it
        if not ctx.options.force_whisper:
            video_id = extract_video_id(ctx.source.url)
            raw = await asyncio.to_thread(self._caption_fetcher, video_id, ctx.options.language)
            if raw:
                ctx.transcript = Transcript(
                    segments=[
                        TranscriptSegment(start_s=r["start"], end_s=r["start"] + r["duration"],
                                          text=r["text"])
                        for r in raw
                    ],
                    language=ctx.options.language,
                    origin=TranscriptOrigin.CAPTIONS,
                )
                return
        path = await asyncio.to_thread(self._audio_downloader, ctx.source.url, self._workdir)
        ctx.audio_path = str(path)

    async def _ingest_local(self, ctx: PipelineContext) -> None:
        src = Path(ctx.source.path or "")
        if not src.exists():
            raise IngestError(f"local file not found: {ctx.source.path}")
        ctx.source = ctx.source.model_copy(update={"title": ctx.source.title or src.stem})
        path = await asyncio.to_thread(self._audio_extractor, str(src), self._workdir)
        ctx.audio_path = str(path)
