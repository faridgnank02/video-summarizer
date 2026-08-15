"""CLI entrypoint: launch the MCP server over stdio or Streamable HTTP."""
from __future__ import annotations

import argparse
import os

from mcp.server.transport_security import TransportSecuritySettings

from .runtime import Runtime
from .server import build_server


def _http_transport_security() -> TransportSecuritySettings:
    """Transport security for the HTTP transport.

    The SDK enables DNS-rebinding protection by default and rejects any Host
    header not in its allowlist -- which rejects every proxied request (nginx
    forwards Host: localhost / the public domain, not the bind address). Set
    MCP_ALLOWED_HOSTS / MCP_ALLOWED_ORIGINS (comma-separated) to keep the
    protection with an explicit allowlist; otherwise disable it, since the
    server is reached only through a reverse proxy that owns Host validation.
    """
    hosts = os.environ.get("MCP_ALLOWED_HOSTS")
    origins = os.environ.get("MCP_ALLOWED_ORIGINS")
    if hosts or origins:
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[h.strip() for h in hosts.split(",")] if hosts else [],
            allowed_origins=[o.strip() for o in origins.split(",")] if origins else [],
        )
    return TransportSecuritySettings(enable_dns_rebinding_protection=False)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.mcp_server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "http":
        server = build_server(Runtime(),
                              transport_security=_http_transport_security())
        server.settings.host = args.host
        server.settings.port = args.port
        server.run(transport="streamable-http")
    else:
        server = build_server(Runtime())
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
