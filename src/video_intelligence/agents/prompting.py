"""Prompt-building helpers shared by LLM-calling agents."""
from __future__ import annotations

from ..schemas import Transcript


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def transcript_lines(transcript: Transcript, block_s: float = 15.0) -> str:
    """Render a transcript as '[M:SS] text' lines, merging segments into ~block_s blocks."""
    lines: list[str] = []
    block_start: float | None = None
    buf: list[str] = []
    for seg in transcript.segments:
        if block_start is None:
            block_start = seg.start_s
        buf.append(seg.text)
        if seg.end_s - block_start >= block_s:
            lines.append(f"[{_ts(block_start)}] {' '.join(buf)}")
            block_start, buf = None, []
    if buf:
        lines.append(f"[{_ts(block_start)}] {' '.join(buf)}")
    return "\n".join(lines)


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most max_chars, breaking on line boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        if size + len(line) + 1 > max_chars and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks
