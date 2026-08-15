# tests/test_schemas.py
from src.video_intelligence.schemas import (
    AnalysisReport, Claim, ClaimVerdict, Evidence, FactCheck, JobOptions, PipelineContext, QualityPreference,
    SourceKind, Transcript, TranscriptOrigin, TranscriptSegment, VideoSource,
    VisualArtifact, VisualKind,
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


def test_visual_artifact_defaults():
    va = VisualArtifact(timestamp_s=12.0, kind=VisualKind.SLIDE)
    assert va.text == ""
    assert va.description is None
    assert va.frame_path is None
    assert va.kind == "slide"


def test_report_visual_highlights_default_empty():
    r = AnalysisReport(summary="s", language="en", trace_id="t")
    assert r.visual_highlights == []


def test_job_options_analyze_visuals_defaults_false():
    assert JobOptions().analyze_visuals is False


def test_pipeline_context_visual_fields_default_none():
    ctx = PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="u"), options=JobOptions())
    assert ctx.video_path is None
    assert ctx.visual_artifacts is None


def test_factcheck_defaults_and_verdict_enum():
    fc = FactCheck(claim="The sky is blue", verdict=ClaimVerdict.SUPPORTED,
                   rationale="Rayleigh scattering.")
    assert fc.verdict == "supported"
    assert fc.evidence == []
    assert fc.search_steps == 0
    assert fc.timestamp_s is None


def test_claim_and_evidence_shapes():
    c = Claim(text="X happened in 2020")
    assert c.timestamp_s is None
    ev = Evidence(title="T", url="https://e.com", snippet="s")
    assert ev.url == "https://e.com"


def test_report_and_options_gain_factcheck_fields():
    report = AnalysisReport(summary="s", language="en", trace_id="t")
    assert report.fact_checks == []
    assert JobOptions().fact_check is False
