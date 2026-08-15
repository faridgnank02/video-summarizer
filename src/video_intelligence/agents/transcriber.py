"""Local speech-to-text via faster-whisper; skipped when captions exist."""
from __future__ import annotations

import asyncio

from ..schemas import PipelineContext, Transcript, TranscriptOrigin, TranscriptSegment
from .base import Agent


class TranscribeError(Exception):
    pass


def _default_model_factory(model_name: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, compute_type="int8")


class Transcriber(Agent):
    name = "transcribe"
    essential = True

    def __init__(self, model_name: str = "base", model_factory=None):
        self._model_name = model_name
        self._model_factory = model_factory or _default_model_factory

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.transcript is not None:
            return ctx  # captions already fetched by the ingestor
        if not ctx.audio_path:
            raise TranscribeError("no audio file and no transcript — ingest must have failed")
        segments, language = await asyncio.to_thread(self._transcribe, ctx.audio_path)
        ctx.transcript = Transcript(segments=segments, language=language,
                                    origin=TranscriptOrigin.WHISPER)
        return ctx

    def _transcribe(self, audio_path: str) -> tuple[list[TranscriptSegment], str]:
        model = self._model_factory(self._model_name)
        raw_segments, info = model.transcribe(audio_path)
        segments = [
            TranscriptSegment(start_s=s.start, end_s=s.end, text=s.text.strip())
            for s in raw_segments
        ]
        return segments, info.language
