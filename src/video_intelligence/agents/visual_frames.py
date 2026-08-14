"""Frame sampling (ffmpeg), OCR (RapidOCR), and text-based classification helpers.

The ffmpeg/OCR functions are the production defaults; agents inject fakes in the
default test suite. classify/normalize/dedup are pure and unit-tested directly.
"""
from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path

from pydantic import BaseModel

from ..schemas import VisualKind


class SampledFrame(BaseModel):
    timestamp_s: float
    image_path: str


_CODE_SYMBOLS = set("{}()[];=<>+/*_#\\|")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_near_duplicate(a: str, b: str, threshold: float = 0.9) -> bool:
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    if not na or not nb:
        return False
    wa, wb = set(na.split()), set(nb.split())
    overlap = len(wa & wb) / max(len(wa | wb), 1)
    return overlap >= threshold


def classify(text: str) -> VisualKind:
    stripped = text.strip()
    if len(stripped) < 15:
        return VisualKind.CHART
    lines = [ln for ln in text.splitlines() if ln.strip()]
    symbol_ratio = sum(c in _CODE_SYMBOLS for c in stripped) / max(len(stripped), 1)
    indented = sum(1 for ln in text.splitlines() if ln[:1] in (" ", "\t"))
    if symbol_ratio > 0.08 or indented >= 2:
        return VisualKind.CODE
    bullets = sum(1 for ln in lines if ln.lstrip()[:1] in ("-", "*", "•"))
    if bullets >= 2 or (len(lines) >= 2 and all(len(ln) < 60 for ln in lines)):
        return VisualKind.SLIDE
    return VisualKind.OTHER


def sample_scene_frames(video_path: str, workdir: str, *, scene_threshold: float,
                        max_frames: int, min_interval_s: float) -> list[SampledFrame]:
    out_dir = Path(workdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex
    pattern = str(out_dir / f"{stem}-%04d.jpg")
    # emit a frame at each scene cut, timestamped via showinfo
    vf = f"select='gt(scene,{scene_threshold})',showinfo"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", vf, "-vsync", "vfr", pattern],
        capture_output=True, text=True,
    )
    times = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]
    files = sorted(out_dir.glob(f"{stem}-*.jpg"))
    frames: list[SampledFrame] = []
    last_t = -1e9
    for i, f in enumerate(files):
        t = times[i] if i < len(times) else float(i)
        if t - last_t < min_interval_s:
            continue
        frames.append(SampledFrame(timestamp_s=t, image_path=str(f)))
        last_t = t
        if len(frames) >= max_frames:
            break
    return frames


def rapidocr_text(image_path: str) -> str:
    from rapidocr_onnxruntime import RapidOCR
    engine = _get_engine()
    result, _ = engine(image_path)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine
