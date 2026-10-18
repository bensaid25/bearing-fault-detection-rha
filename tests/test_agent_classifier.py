import pytest

from agents import classifier_agent


def test_classifier_retries_when_model_skips_the_tool(monkeypatch):
    calls = {"n": 0}

    def fake_chat(model, system_prompt, user_message, tools, tool_functions, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "I think this is a CWRU request."  # no tool call
        tool_functions["submit_classification"](
            pipeline="CWRU", confidence="0.9", reasoning=None
        )
        return ""

    monkeypatch.setattr(classifier_agent, "chat_with_tools", fake_chat)
    result = classifier_agent.classify_pipeline({"rms": 1})
    assert calls["n"] == 2
    assert result.pipeline == "cwru"
    assert result.reasoning == ""


def test_classifier_gives_up_after_all_attempts(monkeypatch):
    monkeypatch.setattr(classifier_agent, "chat_with_tools", lambda **kw: "plain text")
    with pytest.raises(RuntimeError):
        classifier_agent.classify_pipeline({"rms": 1}, attempts=2)