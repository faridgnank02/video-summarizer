from src.video_intelligence.agents.visual_frames import (
    classify, is_near_duplicate, normalize,
)
from src.video_intelligence.schemas import VisualKind


def test_classify_code_by_symbol_and_indent_density():
    code = "def add(a, b):\n    return a + b\n    # sum\nx = add(1, 2)"
    assert classify(code) == VisualKind.CODE


def test_classify_slide_by_title_and_bullets():
    slide = "Roadmap 2026\n- Ship visual agent\n- Fact checker\n- Live streams"
    assert classify(slide) == VisualKind.SLIDE


def test_classify_chart_when_text_sparse():
    assert classify("42%") == VisualKind.CHART
    assert classify("") == VisualKind.CHART


def test_classify_other_for_prose():
    prose = "This is a paragraph of ordinary spoken narration shown on screen."
    assert classify(prose) == VisualKind.OTHER


def test_normalize_collapses_whitespace_and_case():
    assert normalize("  Hello   World \n") == "hello world"


def test_is_near_duplicate_true_for_same_slide():
    assert is_near_duplicate("Roadmap 2026", "roadmap 2026 ") is True


def test_is_near_duplicate_false_for_different_text():
    assert is_near_duplicate("Intro", "Deep dive into caching") is False
