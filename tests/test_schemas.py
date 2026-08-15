# tests/test_schemas.py
from src.video_intelligence.schemas import (
    AnalysisReport, JobOptions, PipelineContext, QualityPreference,
    SourceKind, Transcript, TranscriptOrigin, TranscriptSegment, VideoSource,
)


def test_transcript_full_text_joins_segments():
    t = Transcript(
        segments=[
            TranscriptSegment(start_s=0.0, end_s=2.0, text="Hello"),
            TranscriptSegment(start_s=2.0, end_s=4.0, text="world"),
        ],
        language="en",
        origin=TranscriptOrigin.CAPTIONS,
    )
    assert t.full_text == "Hello world"


def test_job_options_defaults():
    opts = JobOptions()
    assert opts.language == "en"
    assert opts.quality == QualityPreference.BALANCED
    assert opts.force_whisper is False


def test_pipeline_context_generates_trace_id():
    ctx = PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/x"),
        options=JobOptions(),
    )
    assert len(ctx.trace_id) == 32
    assert ctx.transcript is None
    assert ctx.degraded_stages == []


def test_analysis_report_round_trips_json():
    report = AnalysisReport(summary="s", language="en", trace_id="abc")
    parsed = AnalysisReport.model_validate_json(report.model_dump_json())
    assert parsed.summary == "s"
    assert parsed.chapters == []
