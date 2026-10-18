Model: `llama3.1:8b`, 10 runs per payload type (synthetic payloads with the real structure).

| Payload | Correct | Wrong | Errors | Wrong answers | Avg s/run |
|---|---|---|---|---|---|
| cwru | 6/10 | 0 | 4 | - | 18.7 |
| ims | 10/10 | 0 | 0 | - | 5.1 |
| cmapss | 10/10 | 0 | 0 | - | 9.5 |

LLM answers corrected by the rule-based fallback: 4/30. System-level routing accuracy (LLM + fallback): 30/30 for supported payload shapes.

Error causes:
- 1x ResponseError: llama-server process no longer running: exit status 0xc0000409: The system detected an ove
- 3x ResponseError: llama-server process has terminated: exit status 0xc0000409: The system detected an overru
