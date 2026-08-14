import shutil
import subprocess

import pytest

from src.video_intelligence.agents.visualizer import Visualizer
from src.video_intelligence.schemas import (
    JobOptions, PipelineContext, SourceKind, VideoSource,
)

pytestmark = pytest.mark.slow


def _make_slide_video(path: str) -> None:
    # 3s video: white background with the word "ROADMAP" drawn large
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=white:s=640x480:d=3",
        "-vf", "drawtext=text='ROADMAP':fontcolor=black:fontsize=96:x=(w-tw)/2:y=(h-th)/2",
        "-r", "5", path,
    ], check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_visualizer_reads_slide_text(tmp_path):
    video = str(tmp_path / "slide.mp4")
    _make_slide_video(video)
    vis = Visualizer(router=None, scene_threshold=0.0, max_frames=3,
                     min_interval_s=0.0, workdir=str(tmp_path / "frames"))
    ctx = PipelineContext(
        source=VideoSource(kind=SourceKind.LOCAL_FILE, path=video),
        options=JobOptions(analyze_visuals=True), video_path=video)
    ctx = await vis.run(ctx)
    assert ctx.visual_artifacts
    joined = " ".join(a.text.upper() for a in ctx.visual_artifacts)
    assert "ROADMAP" in joined
