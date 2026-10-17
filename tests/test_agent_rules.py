import pytest

from agents.rules import CWRU_KEYS, detect_pipeline_by_rules


def test_rules_cwru():
    assert detect_pipeline_by_rules({k: 0 for k in CWRU_KEYS}) == "cwru"


def test_rules_ims():
    assert detect_pipeline_by_rules({"run": "1st_test", "channels": []}) == "ims"


def test_rules_cmapss():
    assert detect_pipeline_by_rules({"sequence": []}) == "cmapss"


def test_rules_unknown():
    with pytest.raises(ValueError):
        detect_pipeline_by_rules({"foo": 1})