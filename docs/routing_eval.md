Model: `llama3.2:3b`, 20 runs per payload type, up to 3 attempt(s) per call (synthetic payloads with the real structure).

| Payload | Correct | Wrong | Errors | Wrong answers | Avg s/run |
|---|---|---|---|---|---|
| cwru | 20/20 | 0 | 0 | - | 0.9 |
| ims | 20/20 | 0 | 0 | - | 1.3 |
| cmapss | 20/20 | 0 | 0 | - | 1.6 |

LLM answers corrected by the rule-based fallback: 0/60. System-level routing accuracy (LLM + fallback): 60/60 for supported payload shapes.
