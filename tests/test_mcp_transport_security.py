"""HTTP transport security wiring for the MCP server (deployment behind a proxy)."""
from src.mcp_server.__main__ import _http_transport_security


def test_default_disables_dns_rebinding_protection(monkeypatch):
    # Behind nginx/ALB (which owns Host validation) the SDK's default allowlist
    # rejects every proxied Host header, so with no env override we disable it.
    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("MCP_ALLOWED_ORIGINS", raising=False)
    settings = _http_transport_security()
    assert settings.enable_dns_rebinding_protection is False


def test_env_allowlist_enables_protection(monkeypatch):
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "example.com, mcp:8000")
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://example.com")
    settings = _http_transport_security()
    assert settings.enable_dns_rebinding_protection is True
    assert settings.allowed_hosts == ["example.com", "mcp:8000"]
    assert settings.allowed_origins == ["https://example.com"]


def test_build_server_accepts_transport_security():
    from mcp.server.transport_security import TransportSecuritySettings

    from src.mcp_server.runtime import Runtime
    from src.mcp_server.server import build_server

    rt = Runtime(pipeline_factory=lambda on_event=None: None,
                 db_path=":memory:", trace_db=":memory:")
    server = build_server(
        rt, transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False))
    assert server is not None
