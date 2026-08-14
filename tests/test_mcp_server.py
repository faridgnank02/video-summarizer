import pytest

from src.mcp_server.runtime import Runtime
from src.mcp_server.server import build_server
from src.video_intelligence.schemas import AnalysisReport, StageEvent


class FakePipeline:
    def __init__(self, on_event, report):
        self._on_event = on_event
        self._report = report

    async def run(self, source, options):
        if self._on_event is not None:
            await self._on_event(StageEvent(stage="ingest", type="started"))
        return self._report


def make_server(tmp_path):
    report = AnalysisReport(summary="ok", language="en", trace_id="tr1")

    def factory(on_event=None):
        return FakePipeline(on_event, report)

    rt = Runtime(pipeline_factory=factory,
                 db_path=tmp_path / "app.db",
                 trace_db=tmp_path / "traces.db")
    return build_server(rt)


@pytest.mark.asyncio
async def test_server_registers_all_five_tools(tmp_path):
    server = make_server(tmp_path)
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert names == {"analyze_video", "get_job_status", "get_report",
                     "extract_chapters", "get_trace"}


@pytest.mark.asyncio
async def test_analyze_video_tool_input_schema_defaults_quality(tmp_path):
    server = make_server(tmp_path)
    tools = {t.name: t for t in await server.list_tools()}
    schema = tools["analyze_video"].inputSchema
    assert schema["properties"]["quality"]["default"] == "balanced"
    assert "url" in schema["required"]
