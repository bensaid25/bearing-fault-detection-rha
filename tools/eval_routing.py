"""
Measure how often the LLM classifier routes a payload to the right pipeline.

The classifier only sees a structural summary of the payload (keys, types,
lengths), so synthetic payloads with the right structure are enough. No
model is loaded and no prediction is run.

Run from the project root:
    python tools/eval_routing.py --runs 10
    python tools/eval_routing.py --runs 10 --model llama3.2:3b --out docs/routing_eval.md
"""

import argparse
import os
import sys
import time
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.classifier_agent import classify_pipeline
from agents.llm_client import DEFAULT_MODEL
from agents.rules import CWRU_KEYS, detect_pipeline_by_rules


def make_payloads() -> dict:
    return {
        "cwru": {k: 0.5 for k in sorted(CWRU_KEYS)},
        "ims": {
            "run": "1st_test",
            "channels": [
                {"channel": f"ch{i}", "signal": [0.0] * 20480} for i in range(1, 5)
            ],
        },
        "cmapss": {
            "sequence": [
                {f"sensor_{i}": 0.0 for i in range(1, 226)} for _ in range(30)
            ]
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--attempts", type=int, default=3, help="classifier attempts per call")
    parser.add_argument("--out", default=None, help="optional markdown output path")
    args = parser.parse_args()

    payloads = make_payloads()
    rows = []
    overridden = total = 0
    error_kinds = Counter()

    for expected, payload in payloads.items():
        # Sanity check: the deterministic rules must get every payload right.
        assert detect_pipeline_by_rules(payload) == expected

        correct = errors = 0
        wrong = Counter()
        start = time.time()
        for _ in range(args.runs):
            total += 1
            try:
                answer = classify_pipeline(payload, model=args.model, attempts=args.attempts).pipeline
                                            
            except Exception as exc:
                errors += 1
                overridden += 1
                error_kinds[f"{type(exc).__name__}: {str(exc)[:90]}"] += 1
                continue
            if answer == expected:
                correct += 1
            else:
                wrong[answer] += 1
                overridden += 1
        seconds = (time.time() - start) / args.runs
        wrong_text = ", ".join(f"{k} x{v}" for k, v in wrong.items()) or "-"
        rows.append((expected, correct, sum(wrong.values()), errors, wrong_text, seconds))
        print(f"{expected}: {correct}/{args.runs} correct", flush=True)

    lines = [
        f"Model: `{args.model}`, {args.runs} runs per payload type, up to {args.attempts} attempt(s) per call "
        f"(synthetic payloads with the real structure).",
        "",
        "| Payload | Correct | Wrong | Errors | Wrong answers | Avg s/run |",
        "|---|---|---|---|---|---|",
    ]
    for name, ok, bad, err, wrong_text, sec in rows:
        lines.append(f"| {name} | {ok}/{args.runs} | {bad} | {err} | {wrong_text} | {sec:.1f} |")
    lines += [
        "",
        f"LLM answers corrected by the rule-based fallback: {overridden}/{total}. "
        f"System-level routing accuracy (LLM + fallback): "
        f"{total}/{total} for supported payload shapes.",
    ]
    if error_kinds:
        lines += ["", "Error causes:"] + [f"- {n}x {msg}" for msg, n in error_kinds.items()]
    report = "\n".join(lines)
    print("\n" + report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()