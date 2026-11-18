# Eval results (gemini: gemini-3.5-flash-lite)

40 questions: 31 answerable, 9 not in the docs.

| Pipeline | Retrieval hit@5 | Correct answers | Correct refusals | Overall |
| --- | --- | --- | --- | --- |
| baseline (vector only, no guardrail) | 31/31 | 31/31 | 9/9 | 100% |
| improved (hybrid + guardrail) | 31/31 | 30/31 | 9/9 | 98% |

## Misses (improved pipeline)

- **u02** Can I pay with Bitcoin? → "I couldn't find that in our help documents, so I don't want to guess. Would you like me to pass this to a member of our support team?" (The answer failed to state that Bitcoin is not accepted and that Visa is accepted.)
