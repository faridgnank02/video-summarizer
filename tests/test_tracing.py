from src.video_intelligence.schemas import TraceSpan
from src.video_intelligence.tracing import TraceStore


def test_add_and_read_spans_in_order(tmp_path):
    store = TraceStore(tmp_path / "traces.db")
    store.add_span("t1", TraceSpan(stage="transcribe", model_used="whisper-base"))
    store.add_span("t1", TraceSpan(stage="synthesize", model_used="anthropic/claude-sonnet", cost_usd=0.02))
    store.add_span("other", TraceSpan(stage="ingest", model_used="none"))

    spans = store.spans("t1")
    assert [s.stage for s in spans] == ["transcribe", "synthesize"]
    assert spans[1].cost_usd == 0.02


def test_total_cost_sums_only_that_trace(tmp_path):
    store = TraceStore(tmp_path / "traces.db")
    store.add_span("t1", TraceSpan(stage="a", model_used="m", cost_usd=0.01))
    store.add_span("t1", TraceSpan(stage="b", model_used="m", cost_usd=0.02))
    store.add_span("t2", TraceSpan(stage="a", model_used="m", cost_usd=5.0))
    assert store.total_cost("t1") == 0.03


def test_unknown_trace_is_empty(tmp_path):
    store = TraceStore(tmp_path / "traces.db")
    assert store.spans("nope") == []
    assert store.total_cost("nope") == 0.0
