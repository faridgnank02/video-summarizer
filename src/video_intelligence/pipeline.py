"""Async orchestrator: runs agents in order, emits stage events, enforces error policy."""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from .agents.base import Agent
from .schemas import AnalysisReport, JobOptions, PipelineContext, StageEvent, VideoSource

EventCallback = Callable[[StageEvent], Awaitable[None]]


class PipelineError(Exception):
    def __init__(self, stage: str, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(f"{stage}: {reason}")


class Pipeline:
    def __init__(self, agents: list[Agent], on_event: EventCallback | None = None):
        self._agents = agents
        self._on_event = on_event

    async def _emit(self, stage: str, type_: str, message: str | None = None) -> None:
        if self._on_event is not None:
            await self._on_event(StageEvent(stage=stage, type=type_, message=message))

    async def run(self, source: VideoSource, options: JobOptions) -> AnalysisReport:
        ctx = PipelineContext(source=source, options=options)
        for agent in self._agents:
            await self._emit(agent.name, "started")
            try:
                ctx = await agent.run(ctx)
            except Exception as e:
                if agent.essential:
                    await self._emit(agent.name, "failed", str(e))
                    raise PipelineError(agent.name, str(e)) from e
                ctx.degraded_stages.append(agent.name)
                await self._emit(agent.name, "failed", f"degraded: {e}")
                continue
            await self._emit(agent.name, "completed")
        if ctx.report is None:
            raise PipelineError("synthesize", "pipeline finished without a report")
        return ctx.report


def build_pipeline(config_path: str = "config/models.yaml",
                   db_path: str = "data/traces.db",
                   workdir: str = "data/work",
                   on_event: EventCallback | None = None) -> Pipeline:
    """Wire the production pipeline: real providers, router, and agents."""
    from .agents.chapterizer import Chapterizer
    from .agents.ingestor import Ingestor
    from .agents.synthesizer import Synthesizer
    from .agents.transcriber import Transcriber
    from .models.providers.anthropic import AnthropicProvider
    from .models.providers.ollama import OllamaProvider
    from .models.providers.openai import OpenAIProvider
    from .models.router import Router, load_model_config
    from .tracing import TraceStore

    config = load_model_config(config_path)
    store = TraceStore(db_path)
    providers = {
        "ollama": OllamaProvider(),
        "openai": OpenAIProvider(),
        "anthropic": AnthropicProvider(),
    }
    router = Router(config, providers, store)
    whisper_model = config.get("transcription", {}).get("whisper_model", "base")
    return Pipeline(
        [
            Ingestor(workdir=workdir),
            Transcriber(model_name=whisper_model),
            Chapterizer(router),
            Synthesizer(router),
        ],
        on_event=on_event,
    )
