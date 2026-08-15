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


def _split_line(line: str, max_chars: int) -> list[str]:
    """Break a single line into <= max_chars pieces on word boundaries (hard-split a giant word)."""
    if len(line) <= max_chars:
        return [line]
    pieces: list[str] = []
    words = line.split(" ")
    cur = ""
    for word in words:
        # hard-split a single word longer than max_chars
        while len(word) > max_chars:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{cur} {word}".strip()
        if len(candidate) > max_chars and cur:
            pieces.append(cur)
            cur = word
        else:
            cur = candidate
    if cur:
        pieces.append(cur)
    return pieces


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most max_chars, breaking on line then word boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0

    def flush():
        nonlocal current, size
        if current:
            chunks.append("\n".join(current))
            current, size = [], 0

    for line in text.splitlines():
        for piece in _split_line(line, max_chars):
            if size + len(piece) + 1 > max_chars and current:
                flush()
            current.append(piece)
            size += len(piece) + 1
    flush()
    return chunks
