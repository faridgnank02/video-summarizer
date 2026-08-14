"""Frontier-model synthesis: summary, key quotes, action items -> AnalysisReport."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..models.router import Router
from ..schemas import AnalysisReport, KeyQuote, PipelineContext, QualityPreference
from .base import Agent
from .prompting import chunk_text, transcript_lines


class SynthesisResult(BaseModel):
    summary: str
    key_quotes: list[KeyQuote] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)


class PartialSummary(BaseModel):
    summary: str


SYNTH_PROMPT = """You are an analyst producing a structured report on a video.

Video title: <<TITLE>>
Chapters (may be empty): <<CHAPTERS>>

Timestamped transcript (or partial summaries for long videos):
<<TRANSCRIPT>>

Return ONLY a JSON object:
{"summary": "<markdown, 150-400 words>",
 "key_quotes": [{"timestamp_s": <float>, "speaker": null, "text": "<verbatim quote>"}],
 "action_items": ["<string>", ...]}

Rules:
- Write everything in language code: <<LANGUAGE>>.
- 3 to 6 key quotes with timestamps from the [M:SS] markers.
- action_items may be an empty list when the video contains none."""


MAP_PROMPT = """Summarize this portion of a video transcript in under 300 words.
Preserve the most notable statements verbatim with their [M:SS] timestamps.
Return ONLY a JSON object: {"summary": "<string>"}

TRANSCRIPT PORTION:
<<TRANSCRIPT>>"""


class Synthesizer(Agent):
    name = "synthesize"
    essential = True
    MAX_CHARS = 48_000

    def __init__(self, router: Router):
        self._router = router

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        text = transcript_lines(ctx.transcript)
        if len(text) > self.MAX_CHARS:
            text = await self._reduce(ctx, text)
        chapters_str = "; ".join(f"{c.title} ({c.synopsis})" for c in (ctx.chapters or [])) or "none"
        prompt = (SYNTH_PROMPT
                  .replace("<<TITLE>>", ctx.source.title or "unknown")
                  .replace("<<CHAPTERS>>", chapters_str)
                  .replace("<<TRANSCRIPT>>", text)
                  .replace("<<LANGUAGE>>", ctx.options.language))
        result = await self._router.complete(
            task="synthesis", quality=ctx.options.quality, prompt=prompt,
            schema=SynthesisResult, trace_id=ctx.trace_id, stage=self.name,
        )
        ctx.report = AnalysisReport(
            summary=result.summary,
            chapters=ctx.chapters or [],
            key_quotes=result.key_quotes,
            action_items=result.action_items,
            language=ctx.options.language,
            trace_id=ctx.trace_id,
            degraded_stages=list(ctx.degraded_stages),
        )
        return ctx

    async def _reduce(self, ctx: PipelineContext, text: str) -> str:
        parts: list[str] = []
        for chunk in chunk_text(text, self.MAX_CHARS):
            partial = await self._router.complete(
                task="synthesis", quality=QualityPreference.CHEAP,
                prompt=MAP_PROMPT.replace("<<TRANSCRIPT>>", chunk),
                schema=PartialSummary, trace_id=ctx.trace_id, stage="synthesize.map",
            )
            parts.append(partial.summary)
        return "\n\n".join(parts)
