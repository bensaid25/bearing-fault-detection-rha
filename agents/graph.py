import logging
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .classifier_agent import classify_pipeline
from .predictor_agent import run_predictor_agent
from .rules import detect_pipeline_by_rules

logger = logging.getLogger("agents.graph")

CONFIDENCE_THRESHOLD = 0.7


class State(TypedDict, total=False):
    payload: dict
    pipeline: str
    confidence: float
    reasoning: str
    used_fallback: bool
    summary: str


def build_graph(models: dict):
    def classify_node(state: State):
        try:
            c = classify_pipeline(state["payload"])
            return {"pipeline": c.pipeline, "confidence": c.confidence,
                    "reasoning": c.reasoning, "used_fallback": False}
        except Exception as exc:
            logger.warning("LLM classifier failed: %s", exc)
            return {"confidence": 0.0, "reasoning": f"LLM classifier failed: {exc}"}

    def fallback_node(state: State):
        return {"pipeline": detect_pipeline_by_rules(state["payload"]),
                "reasoning": "Rule-based fallback", "used_fallback": True}

    def predict_node(state: State):
        summary = run_predictor_agent(state["pipeline"], state["payload"], models)
        return {"summary": summary}

    def route(state: State) -> Literal["predict", "fallback"]:
        ok = state.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD
        return "predict" if ok else "fallback"

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