"""CLI entrypoint: launch the MCP server over stdio or Streamable HTTP."""
from __future__ import annotations

import argparse

from .runtime import Runtime
from .server import build_server


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.mcp_server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = build_server(Runtime())
    if args.transport == "http":
        server.settings.host = args.host
        server.settings.port = args.port
        server.run(transport="streamable-http")
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
