"""Live rolling-summary agent: streams per-window summaries and produces the final report."""
from __future__ import annotations

from ..live.feed import WindowedTranscriptFeed
from ..live.summarizer import RollingSummarizer
from ..models.router import Router
from ..schemas import PipelineContext
from .base import Agent


class RollingSummarizerAgent(Agent):
    name = "rolling_summarize"
    essential = True

    def __init__(self, router: Router, on_event=None, window_s: int = 60):
        self._router = router
        self._on_event = on_event
        self._window_s = window_s

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.transcript is None:
            raise ValueError("rolling_summarize requires a transcript")
        feed = WindowedTranscriptFeed(ctx.transcript)
        summarizer = RollingSummarizer(self._router, window_s=self._window_s,
                                       quality=ctx.options.quality,
                                       on_event=self._on_event)
        ctx.report = await summarizer.run(feed, ctx.trace_id)
        return ctx
