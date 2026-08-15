from src.video_intelligence.agents.prompting import chunk_text, transcript_lines
from src.video_intelligence.schemas import Transcript, TranscriptOrigin, TranscriptSegment


def make_transcript(n_segments: int, seg_len_s: float = 5.0) -> Transcript:
    segs = [
        TranscriptSegment(start_s=i * seg_len_s, end_s=(i + 1) * seg_len_s, text=f"seg{i}")
        for i in range(n_segments)
    ]
    return Transcript(segments=segs, language="en", origin=TranscriptOrigin.WHISPER)


def test_transcript_lines_merges_into_blocks():
    text = transcript_lines(make_transcript(6))  # 6 x 5s = 30s -> 2 blocks of ~15s
    lines = text.splitlines()
    assert lines[0].startswith("[0:00] ")
    assert "seg0" in lines[0] and "seg2" in lines[0]
    assert lines[1].startswith("[0:15] ")
    assert len(lines) == 2


def test_transcript_lines_formats_hours():
    t = Transcript(
        segments=[TranscriptSegment(start_s=3661.0, end_s=3665.0, text="late")],
        language="en", origin=TranscriptOrigin.WHISPER,
    )
    assert transcript_lines(t).startswith("[1:01:01] ")


def test_chunk_text_respects_line_boundaries():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(10))
    chunks = chunk_text(text, max_chars=200)
    assert all(len(c) <= 200 for c in chunks)
    assert "\n".join(chunks).replace("\n\n", "\n") == text  # nothing lost
    assert len(chunks) > 1


def test_chunk_text_single_chunk_when_small():
    assert chunk_text("short", max_chars=100) == ["short"]


def test_chunk_text_splits_overlong_line():
    line = "word " * 100  # ~500 chars, one line
    chunks = chunk_text(line, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) > 1


def test_chunk_text_hard_splits_giant_word():
    chunks = chunk_text("x" * 500, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
