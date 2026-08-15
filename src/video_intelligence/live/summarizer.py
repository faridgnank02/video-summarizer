"""Rolling window summarization: per-window delta + cumulative digest + final report."""
from __future__ import annotations

from pydantic import BaseModel

from ..agents.prompting import transcript_lines
from ..models.router import Router, RouterError
from ..schemas import (
    AnalysisReport, QualityPreference, RollingSummary, StageEvent, Transcript,
    TranscriptOrigin, TranscriptSegment,
)
from .feed import SegmentFeed


class RollingResult(BaseModel):
    summary: str


DELTA_PROMPT = """Summarize what is NEW in this portion of a live video transcript, in 1-2 sentences.
Return ONLY a JSON object: {"summary": "<string>"}
Write in language code: <<LANGUAGE>>.

TRANSCRIPT PORTION:
<<TEXT>>"""


FOLD_PROMPT = """You maintain a running digest of a live video. Merge the NEW UPDATE into the
CURRENT DIGEST, keeping it concise and in chronological order.
Return ONLY a JSON object: {"summary": "<updated running digest>"}
Write in language code: <<LANGUAGE>>.

CURRENT DIGEST:
<<DIGEST>>

NEW UPDATE:
<<DELTA>>"""


CONSOLIDATE_PROMPT = """Produce a final, well-structured summary of a video from its running digest.
Return ONLY a JSON object: {"summary": "<markdown, 120-300 words>"}
Write in language code: <<LANGUAGE>>.

RUNNING DIGEST:
<<DIGEST>>"""


class RollingSummarizer:
    def __init__(self, router: Router, window_s: int = 60,
                 quality: QualityPreference = QualityPreference.BALANCED,
                 on_event=None):
        self._router = router
        self._window_s = window_s
        self._quality = quality
        self._on_event = on_event
        self._language = "en"

    async def run(self, feed: SegmentFeed, trace_id: str) -> AnalysisReport:
        self._language = feed.language
        window: list[TranscriptSegment] = []
        digest = ""
        window_index = 0
        async for seg in feed.segments():
            window.append(seg)
            if window[-1].end_s - window[0].start_s >= self._window_s:
                digest = await self._close_window(window, window_index, digest, trace_id)
                window_index += 1
                window = []
        if window:
            digest = await self._close_window(window, window_index, digest, trace_id)
        summary = await self._consolidate(digest, trace_id) if digest else ""
        return AnalysisReport(summary=summary, language=self._language, trace_id=trace_id)

    async def _close_window(self, segs: list[TranscriptSegment], index: int,
                            prev_digest: str, trace_id: str) -> str:
        text = transcript_lines(Transcript(segments=segs, language=self._language,
                                           origin=TranscriptOrigin.CAPTIONS))
        try:
            delta = await self._complete(DELTA_PROMPT.replace("<<TEXT>>", text),
                                         "rolling.delta", trace_id)
            digest = delta if not prev_digest else await self._complete(
                FOLD_PROMPT.replace("<<DIGEST>>", prev_digest).replace("<<DELTA>>", delta),
                "rolling.fold", trace_id)
        except RouterError:
            delta = "(summary unavailable for this window)"
            digest = prev_digest
        await self._emit(RollingSummary(
            window_index=index, window_start_s=segs[0].start_s,
            window_end_s=segs[-1].end_s, delta=delta, running_summary=digest))
        return digest

    async def _consolidate(self, digest: str, trace_id: str) -> str:
        try:
            return await self._complete(CONSOLIDATE_PROMPT.replace("<<DIGEST>>", digest),
                                        "rolling.consolidate", trace_id)
        except RouterError:
            return digest

    async def _complete(self, prompt: str, stage: str, trace_id: str) -> str:
        result = await self._router.complete(
            task="rolling", quality=self._quality,
            prompt=prompt.replace("<<LANGUAGE>>", self._language),
            schema=RollingResult, trace_id=trace_id, stage=stage)
        return result.summary

    async def _emit(self, rs: RollingSummary) -> None:
        if self._on_event is not None:
            await self._on_event(StageEvent(stage="live", type="summary",
                                            message=rs.running_summary,
                                            data=rs.model_dump()))
