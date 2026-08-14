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
    quality_prop = schema["properties"]["quality"]
    defs = schema.get("$defs", {})

    def find_enum(node):
        if not isinstance(node, dict):
            return None
        if "enum" in node:
            return node["enum"]
        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            return find_enum(defs.get(ref_name, {}))
        for key in ("allOf", "anyOf", "oneOf"):
            for sub in node.get(key, []):
                found = find_enum(sub)
                if found:
                    return found
        return None

    enum_values = find_enum(quality_prop)
    assert enum_values is not None, quality_prop
    assert set(enum_values) == {"cheap", "balanced", "best"}


@pytest.mark.asyncio
async def test_analyze_video_exposes_analyze_visuals(tmp_path):
    server = make_server(tmp_path)
    tools = {t.name: t for t in await server.list_tools()}
    assert "analyze_visuals" in tools["analyze_video"].inputSchema["properties"]


@pytest.mark.asyncio
async def test_extract_chapters_exposes_analyze_visuals(tmp_path):
    server = make_server(tmp_path)
    tools = {t.name: t for t in await server.list_tools()}
    assert "analyze_visuals" in tools["extract_chapters"].inputSchema["properties"]
