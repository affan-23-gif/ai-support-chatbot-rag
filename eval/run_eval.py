"""Compare a naive RAG baseline with the improved pipeline on eval/questions.jsonl.

  baseline : vector search only, no guardrail (always answers)
  improved : hybrid search (vector + BM25) + "not in the docs" guardrail

Metrics
  retrieval hit@5     : the expected document is among the 5 retrieved chunks (answerable questions)
  correct answers     : an LLM judge checks the answer against the expected key facts (answerable)
  correct refusals    : the bot declined instead of inventing an answer (unanswerable questions)

Usage:  python -m eval.run_eval            (uses .env, i.e. Gemini)
        LLM_PROVIDER=fake python -m eval.run_eval   (offline smoke test; numbers are NOT meaningful)
Writes eval/results.md and eval/results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import SupportAgent  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import Store  # noqa: E402
from app.ingest import load_directory  # noqa: E402
from app.llm import get_llm  # noqa: E402
from app.retrieval import HybridIndex  # noqa: E402

HERE = Path(__file__).resolve().parent

JUDGE_SYSTEM = """TASK: JUDGE
You grade a customer-support answer. The expected key facts are separated by "|".
The answer is correct if it states the key facts (wording may differ) and does not contradict them.
Return JSON: {"correct": true | false, "reason": "<short reason>"}"""

REFUSAL_SYSTEM = """TASK: REFUSAL_JUDGE
The question below cannot be answered from the store's documents.
Did the assistant decline or say it doesn't know, instead of giving a specific made-up answer?
Return JSON: {"refused": true | false, "reason": "<short reason>"}"""

CONFIGS = {
    "baseline (vector only, no guardrail)": {"mode": "vector", "strict": False},
    "improved (hybrid + guardrail)": {"mode": "hybrid", "strict": True},
}


def main() -> None:
    questions = [json.loads(line) for line in (HERE / "questions.jsonl").read_text().splitlines() if line.strip()]
    llm = get_llm()
    index = HybridIndex(llm, HERE / ".index")  # separate copy, so the running server's index isn't locked
    index.build(load_directory(settings.docs_dir, settings.chunk_size, settings.chunk_overlap))
    agent = SupportAgent(llm, index, Store(HERE / ".eval.db"))

    if settings.llm_provider != "fake":
        calls = 2 * len(CONFIGS) * len(questions)
        print(f"About {calls} Gemini calls at {settings.gemini_rpm}/min ≈ {calls // settings.gemini_rpm + 1} minutes. "
              "Leave it running.")

    summary, details = {}, []
    for name, cfg in CONFIGS.items():
        hit = correct = refused = 0
        n_ans = sum(q["answerable"] for q in questions)
        n_un = len(questions) - n_ans
        for q in questions:
            reply, hits = agent.answer_question(q["question"], mode=cfg["mode"], strict=cfg["strict"])
            row = {"config": name, "id": q["id"], "question": q["question"], "answer": reply.answer,
                   "action": reply.action, "retrieved": [h.chunk.source for h in hits]}
            if q["answerable"]:
                row["hit"] = q["source"] in row["retrieved"]
                verdict = llm.generate_json(JUDGE_SYSTEM, f"<question>\n{q['question']}\n</question>\n"
                                            f"<expected>\n{q['expected']}\n</expected>\n<answer>\n{reply.answer}\n</answer>")
                row["correct"], row["reason"] = bool(verdict.get("correct")), verdict.get("reason", "")
                hit += row["hit"]
                correct += row["correct"]
            else:
                if reply.action == "not_found":
                    row["refused"], row["reason"] = True, "guardrail"
                else:
                    verdict = llm.generate_json(REFUSAL_SYSTEM, f"<question>\n{q['question']}\n</question>\n"
                                                f"<answer>\n{reply.answer}\n</answer>")
                    row["refused"], row["reason"] = bool(verdict.get("refused")), verdict.get("reason", "")
                refused += row["refused"]
            details.append(row)
            ok = row.get("correct", row.get("refused"))
            print(f"  [{len(details) % len(questions) or len(questions):>2}/{len(questions)}] "
                  f"{'ok  ' if ok else 'MISS'} {q['id']} {q['question'][:60]}", flush=True)
        summary[name] = {"hit@5": f"{hit}/{n_ans}", "correct answers": f"{correct}/{n_ans}",
                         "correct refusals": f"{refused}/{n_un}",
                         "overall": f"{round(100 * (correct + refused) / len(questions))}%"}
        print(name, summary[name])

    (HERE / "results.json").write_text(json.dumps({"summary": summary, "details": details}, indent=1))
    lines = [f"# Eval results ({settings.llm_provider}: {settings.chat_model if settings.llm_provider != 'fake' else 'offline fake'})",
             "", f"{len(questions)} questions: {sum(q['answerable'] for q in questions)} answerable, "
             f"{sum(not q['answerable'] for q in questions)} not in the docs.", "",
             "| Pipeline | Retrieval hit@5 | Correct answers | Correct refusals | Overall |",
             "| --- | --- | --- | --- | --- |"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['hit@5']} | {s['correct answers']} | {s['correct refusals']} | {s['overall']} |")
    lines += ["", "## Misses (improved pipeline)", ""]
    for r in details:
        if r["config"].startswith("improved") and not (r.get("correct") or r.get("refused")):
            lines.append(f"- **{r['id']}** {r['question']} → {r['answer'][:160]!r} ({r.get('reason', '')})")
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    index.close()
    print("Wrote eval/results.md")


if __name__ == "__main__":
    main()
