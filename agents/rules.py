"""agents/rules.py: deterministic payload routing (no heavy imports)."""

CWRU_KEYS = {
    "rms", "kurtosis", "skewness", "peak_to_peak", "std", "dominant_freq",
    "spectral_energy", "spectral_centroid", "energy_0_1000",
    "energy_1000_2500", "energy_2500_5000", "load",
}


def detect_pipeline_by_rules(payload: dict) -> str:
    """Decide the pipeline from the payload's keys alone."""
    if "sequence" in payload:
        return "cmapss"
    if "run" in payload and "channels" in payload:
        return "ims"
    if CWRU_KEYS.issubset(payload):
        return "cwru"
    raise ValueError("Unrecognised payload shape")