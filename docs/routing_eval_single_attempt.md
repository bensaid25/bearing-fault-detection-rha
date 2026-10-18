Model: `llama3.2:3b`, 20 runs per payload type, up to 1 attempt(s) per call (synthetic payloads with the real structure).

| Payload | Correct | Wrong | Errors | Wrong answers | Avg s/run |
|---|---|---|---|---|---|
| cwru | 19/20 | 0 | 1 | - | 1.7 |
| ims | 12/20 | 0 | 8 | - | 1.3 |
| cmapss | 20/20 | 0 | 0 | - | 1.4 |

LLM answers corrected by the rule-based fallback: 9/60. System-level routing accuracy (LLM + fallback): 60/60 for supported payload shapes.

Error causes:
- 9x RuntimeError: No valid classification after 1 attempts; last reply was '{"name": "submit_classification"
