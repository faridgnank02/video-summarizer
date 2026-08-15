"""MCP adapter over the video_intelligence pipeline."""
from .runtime import Runtime
from .server import build_server

__all__ = ["Runtime", "build_server"]
