import pytest

from src.video_intelligence.agents.visualizer import Visualizer
from src.video_intelligence.agents.visual_frames import SampledFrame
from src.video_intelligence.schemas import (
    JobOptions, PipelineContext, SourceKind, VideoSource, VisualKind,
)


def ctx_for(video_path=None, **opts):
    return PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="u"),
        options=JobOptions(**opts), video_path=video_path)


def make(frames, ocr_map):
    def sampler(video_path, workdir, *, scene_threshold, max_frames, min_interval_s):
        return frames

    def ocr(image_path):
        return ocr_map.get(image_path, "")

    return Visualizer(router=None, frame_sampler=sampler, ocr=ocr)


async def test_skips_when_analyze_visuals_off():
    vis = make([], {})
    ctx = await make([], {}).run(ctx_for(video_path="v.mp4", analyze_visuals=False))
    assert ctx.visual_artifacts is None


async def test_raises_when_visuals_on_but_no_video():
    vis = make([], {})
    with pytest.raises(ValueError, match="requires a video"):
        await vis.run(ctx_for(video_path=None, analyze_visuals=True))


async def test_produces_classified_artifacts():
    frames = [
        SampledFrame(timestamp_s=5.0, image_path="a.jpg"),
        SampledFrame(timestamp_s=20.0, image_path="b.jpg"),
    ]
    ocr_map = {"a.jpg": "def f():\n    return 1", "b.jpg": "Roadmap\n- one\n- two"}
    vis = make(frames, ocr_map)
    ctx = await vis.run(ctx_for(video_path="v.mp4", analyze_visuals=True))
    kinds = [a.kind for a in ctx.visual_artifacts]
    assert VisualKind.CODE in kinds and VisualKind.SLIDE in kinds
    assert ctx.visual_artifacts[0].timestamp_s == 5.0
    assert ctx.visual_artifacts[0].frame_path == "a.jpg"


async def test_dedups_consecutive_near_identical_frames():
    frames = [
        SampledFrame(timestamp_s=5.0, image_path="a.jpg"),
        SampledFrame(timestamp_s=6.0, image_path="b.jpg"),
    ]
    ocr_map = {"a.jpg": "Roadmap 2026", "b.jpg": "roadmap 2026 "}
    vis = make(frames, ocr_map)
    ctx = await vis.run(ctx_for(video_path="v.mp4", analyze_visuals=True))
    assert len(ctx.visual_artifacts) == 1
