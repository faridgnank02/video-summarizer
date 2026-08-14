"""Execution runtime for the MCP adapter — framework-free, testable without MCP."""
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from src.api.jobs import JobStore
from src.video_intelligence.pipeline import PipelineError, build_pipeline
from src.video_intelligence.schemas import (
    JobOptions, QualityPreference, SourceKind, StageEvent, VideoSource,
)
from src.video_intelligence.tracing import TraceStore

EventCallback = Callable[[StageEvent], Awaitable[None]]


async def _noop(_ev: StageEvent) -> None:
    return None


class Runtime:
    def __init__(self, pipeline_factory=build_pipeline,
                 db_path: str | Path = "data/app.db",
                 trace_db: str | Path = "data/traces.db"):
        self.factory = pipeline_factory
        self.jobs = JobStore(db_path)
        self.traces = TraceStore(trace_db)

    async def _execute(self, job_id: str, source: VideoSource,
                       options: JobOptions, on_event: EventCallback) -> dict:
        pipeline = self.factory(on_event=on_event)
        self.jobs.update(job_id, status="running")
        report = await pipeline.run(source, options)
        self.jobs.update(job_id, status="completed", report=report,
                         trace_id=report.trace_id)
        return {"status": "completed", "job_id": job_id,
                "trace_id": report.trace_id, "report": report.model_dump()}

    async def analyze(self, url: str, quality: str = "balanced",
                      language: str = "en", force_whisper: bool = False,
                      async_: bool = False,
                      on_event: EventCallback | None = None) -> dict:
        source = VideoSource(kind=SourceKind.YOUTUBE, url=url)
        options = JobOptions(language=language,
                             quality=QualityPreference(quality),
                             force_whisper=force_whisper)
        job_id = uuid.uuid4().hex
        self.jobs.create(job_id, source, options)
        return await self._execute(job_id, source, options, on_event or _noop)
