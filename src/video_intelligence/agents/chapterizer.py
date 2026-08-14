"""Timestamped chapter segmentation via a cheap model."""
from __future__ import annotations

from pydantic import BaseModel

from ..models.router import Router
from ..schemas import Chapter, PipelineContext
from .base import Agent
from .prompting import chunk_text, transcript_lines


class ChapterList(BaseModel):
    chapters: list[Chapter]


CHAPTER_PROMPT = """You segment video transcripts into chapters.

Given the timestamped transcript below, return ONLY a JSON object:
{"chapters": [{"start_s": <float>, "end_s": <float>, "title": "<string>", "synopsis": "<string>"}]}

Rules:
- 3 to 12 chapters covering the whole transcript, in order, non-overlapping.
- Titles under 8 words; synopsis is one sentence.
- start_s / end_s are seconds, taken from the [M:SS] timestamps.

TRANSCRIPT:
<<TRANSCRIPT>>"""


class Chapterizer(Agent):
    name = "chapterize"
    essential = False
    MAX_CHARS = 24_000

    def __init__(self, router: Router):
        self._router = router

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        text = transcript_lines(ctx.transcript)
        chapters: list[Chapter] = []
        for chunk in chunk_text(text, self.MAX_CHARS):
            result = await self._router.complete(
                task="chaptering",
                quality=ctx.options.quality,
                prompt=CHAPTER_PROMPT.replace("<<TRANSCRIPT>>", chunk),
                schema=ChapterList,
                trace_id=ctx.trace_id,
                stage=self.name,
            )
            chapters.extend(result.chapters)
        ctx.chapters = chapters
        return ctx
