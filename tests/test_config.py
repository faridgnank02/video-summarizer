"""Guards against regressing config/models.yaml's routing setup."""
from src.video_intelligence.models.router import load_model_config

CONFIG_PATH = "config/models.yaml"


def test_synthesis_cheap_tier_spans_multiple_providers():
    config = load_model_config(CONFIG_PATH)
    cheap = config["tasks"]["synthesis"]["cheap"]
    assert len(cheap) >= 2
    providers = {candidate.split("/", 1)[0] for candidate in cheap}
    assert len(providers) >= 2
