"""Visual agent: sample frames -> OCR -> classify -> VisualArtifacts (non-essential)."""
from __future__ import annotations

from ..schemas import PipelineContext, VisualArtifact
from .base import Agent
from .visual_frames import classify, is_near_duplicate, rapidocr_text, sample_scene_frames


class Visualizer(Agent):
    name = "visual"
    essential = False

    def __init__(self, router, *, frame_sampler=sample_scene_frames, ocr=rapidocr_text,
                 scene_threshold: float = 0.4, max_frames: int = 24,
                 min_interval_s: float = 8.0, workdir: str = "data/work"):
        self._router = router
        self._frame_sampler = frame_sampler
        self._ocr = ocr
        self._scene_threshold = scene_threshold
        self._max_frames = max_frames
        self._min_interval_s = min_interval_s
        self._workdir = workdir

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.options.analyze_visuals:
            return ctx
        if ctx.video_path is None:
            raise ValueError("visual requires a video")
        frames = self._frame_sampler(
            ctx.video_path, self._workdir, scene_threshold=self._scene_threshold,
            max_frames=self._max_frames, min_interval_s=self._min_interval_s)
        artifacts: list[VisualArtifact] = []
        last_text = ""
        for frame in frames:
            text = self._ocr(frame.image_path)
            if text and last_text and is_near_duplicate(text, last_text):
                continue
            last_text = text
            artifacts.append(VisualArtifact(
                timestamp_s=frame.timestamp_s, kind=classify(text),
                text=text, frame_path=frame.image_path))
        ctx.visual_artifacts = artifacts
        return ctx
