import logging
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .classifier_agent import classify_pipeline
from .predictor_agent import format_summary, run_prediction
from .rules import detect_pipeline_by_rules

logger = logging.getLogger("agents.graph")


class State(TypedDict, total=False):
    payload: dict
    llm_pipeline: str   # what the LLM answered (kept for evaluation)
    pipeline: str       # the pipeline actually used
    confidence: float
    reasoning: str
    used_fallback: bool
    prediction: dict
    summary: str
    error: str


def build_graph(models: dict):
    def classify_node(state: State):
        try:
            c = classify_pipeline(state["payload"])
        except Exception as exc:
            logger.warning("LLM classifier failed: %s", exc)
            return {"reasoning": f"LLM classifier failed: {exc}"}
        return {"llm_pipeline": c.pipeline, "pipeline": c.pipeline,
                "confidence": c.confidence, "reasoning": c.reasoning,
                "used_fallback": False}

    def fallback_node(state: State):
        return {"pipeline": detect_pipeline_by_rules(state["payload"]),
                "reasoning": "Rule-based fallback", "used_fallback": True}

    def predict_node(state: State):
        try:
            result = run_prediction(state["pipeline"], state["payload"], models)
        except Exception as exc:
            return {"error": f"Prediction failed: {exc}"}
        return {"prediction": result,
                "summary": format_summary(state["pipeline"], result)}

    def route(state: State) -> Literal["predict", "fallback"]:
        # Small LLMs report meaningless confidence values, so trust the LLM
        # only when it agrees with the deterministic key-based rules.
        try:
            expected = detect_pipeline_by_rules(state["payload"])
        except ValueError:
            return "fallback"  # unknown payload shape: fallback raises a clear error
        return "predict" if state.get("llm_pipeline") == expected else "fallback"

    g = StateGraph(State)
    g.add_node("classify", classify_node)
    g.add_node("fallback", fallback_node)
    g.add_node("predict", predict_node)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route, {"predict": "predict", "fallback": "fallback"})
    g.add_edge("fallback", "predict")
    g.add_edge("predict", END)
    return g.compile()


if __name__ == "__main__":
    print(build_graph(models={}).get_graph().draw_mermaid())