"""Visual agent: sample frames -> OCR -> classify -> VisualArtifacts (non-essential)."""
from __future__ import annotations

from pydantic import BaseModel

from ..models.router import RouterError
from ..schemas import PipelineContext, QualityPreference, VisualArtifact, VisualKind
from .base import Agent
from .visual_frames import classify, is_near_duplicate, rapidocr_text, sample_scene_frames


class VisionDescription(BaseModel):
    description: str


_VISION_PROMPT = (
    "Describe the meaningful visual content of this video frame in one or two "
    "sentences (a chart, diagram, or figure). Return ONLY JSON: "
    '{"description": "<string>"}'
)


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
        if ctx.options.quality == QualityPreference.BEST and self._router is not None:
            for art in artifacts:
                if art.kind != VisualKind.CHART:
                    continue
                try:
                    with open(art.frame_path, "rb") as fh:
                        image = fh.read()
                    result = await self._router.complete_vision(
                        task="visual_description", quality=ctx.options.quality,
                        prompt=_VISION_PROMPT, images=[image], schema=VisionDescription,
                        trace_id=ctx.trace_id, stage=self.name)
                    art.description = result.description
                except (RouterError, OSError):
                    continue  # vision unavailable/unreadable: OCR text stands
        ctx.visual_artifacts = artifacts
        return ctx
