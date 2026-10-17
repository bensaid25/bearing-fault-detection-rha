import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.graph import build_graph
from api.model_loader import load_all_models

cwru_payload = {
    "rms": 0.42, "kurtosis": 3.1, "skewness": 0.05, "peak_to_peak": 1.8,
    "std": 0.39, "dominant_freq": 118.5, "spectral_energy": 22.7,
    "spectral_centroid": 340.2, "energy_0_1000": 5.1,
    "energy_1000_2500": 3.4, "energy_2500_5000": 1.2, "load": 1,
}

models = load_all_models()
graph = build_graph(models)
out = graph.invoke({"payload": cwru_payload})

print("Pipeline:", out["pipeline"])
print("Confidence:", out.get("confidence"))
print("Used fallback:", out["used_fallback"])
print("Summary:", out["summary"])