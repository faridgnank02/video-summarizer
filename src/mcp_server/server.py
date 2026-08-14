"""FastMCP tool surface bridging MCP calls to the Runtime."""
from __future__ import annotations

from mcp.server.fastmcp import Context, FastMCP

from src.video_intelligence.schemas import StageEvent

from .runtime import Runtime


def build_server(runtime: Runtime) -> FastMCP:
    mcp = FastMCP("Video Intelligence")

    def _progress(ctx: Context):
        async def on_event(ev: StageEvent) -> None:
            await ctx.info(f"{ev.stage}: {ev.type}"
                           + (f" - {ev.message}" if ev.message else ""))
        return on_event

    @mcp.tool()
    async def analyze_video(url: str, ctx: Context, quality: str = "balanced",
                            language: str = "en", force_whisper: bool = False,
                            async_: bool = False) -> dict:
        """Analyze a YouTube video into summary, chapters, quotes, and action items."""
        return await runtime.analyze(
            url=url, quality=quality, language=language,
            force_whisper=force_whisper, async_=async_,
            on_event=_progress(ctx))

    @mcp.tool()
    async def get_job_status(job_id: str) -> dict:
        """Status of an async analyze_video job."""
        return runtime.job_status(job_id)

    @mcp.tool()
    async def get_report(job_id: str) -> dict:
        """The analysis report for a completed job."""
        return runtime.get_report(job_id)

    @mcp.tool()
    async def extract_chapters(url: str, ctx: Context, quality: str = "balanced",
                               language: str = "en") -> list | dict:
        """Timestamped chapters for a YouTube video."""
        return await runtime.extract_chapters(
            url=url, quality=quality, language=language, on_event=_progress(ctx))

    @mcp.tool()
    async def get_trace(trace_id: str) -> dict:
        """Cost and latency spans for a completed analysis."""
        return runtime.get_trace(trace_id)

    return mcp
