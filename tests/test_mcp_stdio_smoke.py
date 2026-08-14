import pytest

mcp_client = pytest.importorskip("mcp.client.stdio")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


@pytest.mark.slow
@pytest.mark.asyncio
async def test_stdio_handshake_lists_all_tools(tmp_path):
    params = StdioServerParameters(
        command="python3",
        args=["-m", "src.mcp_server", "--transport", "stdio"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
    names = {t.name for t in listed.tools}
    assert names == {"analyze_video", "get_job_status", "get_report",
                     "extract_chapters", "get_trace"}
