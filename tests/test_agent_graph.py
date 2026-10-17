from agents import graph as graph_module
from agents.classifier_agent import ClassificationResult
from agents.rules import CWRU_KEYS


def build(monkeypatch, classifier):
    monkeypatch.setattr(graph_module, "classify_pipeline", classifier)
    monkeypatch.setattr(
        graph_module, "run_prediction",
        lambda pipeline, payload, models: {"label": pipeline},
    )
    monkeypatch.setattr(
        graph_module, "format_summary",
        lambda pipeline, result: f"summary for {pipeline}",
    )
    return graph_module.build_graph(models={})


def test_confident_llm_is_used(monkeypatch):
    g = build(monkeypatch, lambda payload: ClassificationResult(
        pipeline="cwru", confidence=0.95, reasoning="ok"))
    out = g.invoke({"payload": {k: 0 for k in CWRU_KEYS}})
    assert out["pipeline"] == "cwru"
    assert out["used_fallback"] is False
    assert out["summary"] == "summary for cwru"


def test_low_confidence_triggers_fallback(monkeypatch):
    # The LLM guesses wrong and is unsure; the rules override it.
    g = build(monkeypatch, lambda payload: ClassificationResult(
        pipeline="ims", confidence=0.3, reasoning="unsure"))
    out = g.invoke({"payload": {"sequence": []}})
    assert out["pipeline"] == "cmapss"
    assert out["used_fallback"] is True


def test_llm_failure_triggers_fallback(monkeypatch):
    def broken(payload):
        raise RuntimeError("Ollama is down")

    g = build(monkeypatch, broken)
    out = g.invoke({"payload": {"run": "1st_test", "channels": []}})
    assert out["pipeline"] == "ims"
    assert out["used_fallback"] is True


def test_confident_but_wrong_llm_is_overridden(monkeypatch):
    g = build(monkeypatch, lambda payload: ClassificationResult(
        pipeline="ims", confidence=0.95, reasoning="sure"))
    out = g.invoke({"payload": {"sequence": []}})
    assert out["pipeline"] == "cmapss"
    assert out["used_fallback"] is True


def test_failed_prediction_is_reported_not_summarized(monkeypatch):
    g = build(monkeypatch, lambda p: ClassificationResult(
        pipeline="cwru", confidence=0.95, reasoning="ok"))

    def failing(pipeline, payload, models):
        raise RuntimeError("tool crashed")

    monkeypatch.setattr(graph_module, "run_prediction", failing)
    out = g.invoke({"payload": {k: 0 for k in CWRU_KEYS}})
    assert "summary" not in out
    assert "tool crashed" in out["error"]

def test_agreeing_llm_is_accepted_despite_zero_confidence(monkeypatch):
    g = build(monkeypatch, lambda payload: ClassificationResult(
        pipeline="cwru", confidence=0.0, reasoning=""))
    out = g.invoke({"payload": {k: 0 for k in CWRU_KEYS}})
    assert out["used_fallback"] is False
    assert out["summary"] == "summary for cwru"