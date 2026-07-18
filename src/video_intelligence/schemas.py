"""Shared data shapes for the video-intelligence pipeline."""
from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field


class SourceKind(StrEnum):
    YOUTUBE = "youtube"
    LOCAL_FILE = "local_file"


class TranscriptOrigin(StrEnum):
    CAPTIONS = "captions"
    WHISPER = "whisper"


class QualityPreference(StrEnum):
    CHEAP = "cheap"
    BALANCED = "balanced"
    BEST = "best"


class VideoSource(BaseModel):
    kind: SourceKind
    url: str | None = None
    path: str | None = None
    title: str | None = None
    duration_s: float | None = None
    channel: str | None = None


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    text: str


class Transcript(BaseModel):
    segments: list[TranscriptSegment]
    language: str
    origin: TranscriptOrigin

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)


class Chapter(BaseModel):
    start_s: float
    end_s: float
    title: str
    synopsis: str


class KeyQuote(BaseModel):
    timestamp_s: float
    speaker: str | None = None
    text: str


class AnalysisReport(BaseModel):
    summary: str
    chapters: list[Chapter] = Field(default_factory=list)
    key_quotes: list[KeyQuote] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    language: str
    trace_id: str
    degraded_stages: list[str] = Field(default_factory=list)


class TraceSpan(BaseModel):
    stage: str
    model_used: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    status: str = "ok"  # ok | error
    fallback_from: str | None = None


class JobOptions(BaseModel):
    language: str = "en"
    quality: QualityPreference = QualityPreference.BALANCED
    force_whisper: bool = False


class StageEvent(BaseModel):
    stage: str
    type: str  # started | progress | completed | failed
    message: str | None = None


class PipelineContext(BaseModel):
    source: VideoSource
    options: JobOptions
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    audio_path: str | None = None
    transcript: Transcript | None = None
    chapters: list[Chapter] | None = None
    report: AnalysisReport | None = None
    degraded_stages: list[str] = Field(default_factory=list)
