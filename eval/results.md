# Eval results (gemini: gemini-3.5-flash-lite)

40 questions: 30 answerable, 10 not in the docs.

| Pipeline | Retrieval hit@5 | Correct answers | Correct refusals | Overall |
| --- | --- | --- | --- | --- |
| baseline (vector only, no guardrail) | 30/30 | 28/30 | 9/10 | 92% |
| improved (hybrid + guardrail) | 30/30 | 30/30 | 10/10 | 100% |

## Misses (improved pipeline)

