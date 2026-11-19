"""Compare a naive RAG baseline with the improved pipeline on eval/questions.jsonl.

  baseline : vector search only, no guardrail (always answers)
  improved : hybrid search (vector + BM25) + "not in the docs" guardrail

Metrics
  retrieval hit@5     : the expected document is among the 5 retrieved chunks (answerable questions)
  correct answers     : an LLM judge checks the answer against the expected key facts (answerable)
  correct refusals    : the bot declined instead of inventing an answer (unanswerable questions)

Usage:  python -m eval.run_eval            (uses .env, i.e. Gemini)
        python -m eval.run_eval --fresh    (ignore saved results and re-run everything)
        LLM_PROVIDER=fake python -m eval.run_eval   (offline smoke test; numbers are NOT meaningful)

Every graded question is saved to eval/.cache.json right away. If a run stops (quota,
network, Ctrl+C), run it again: finished questions are reused and it continues where it
stopped. The cache key includes the prompts, models, settings and documents, so any change
there re-runs the affected questions automatically.
Writes eval/results.md and eval/results.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import agent as agent_module  # noqa: E402
from app.agent import SupportAgent  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import Store  # noqa: E402
from app.ingest import load_directory  # noqa: E402
from app.llm import DailyQuotaExceeded, get_llm  # noqa: E402
from app.retrieval import HybridIndex  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache.json"

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


def run_fingerprint() -> str:
    """Everything that can change an answer or a grade. Changing any of it invalidates the cache."""
    docs = sorted((p.name, hashlib.sha1(p.read_bytes()).hexdigest()) for p in settings.docs_dir.iterdir()
                  if p.is_file())
    parts = [settings.llm_provider, settings.chat_model, settings.judge_model, settings.embedding_model,
             settings.chunk_size, settings.chunk_overlap, settings.top_k, settings.min_similarity,
             agent_module.ANSWER_SYSTEM, agent_module.LENIENT_SYSTEM, JUDGE_SYSTEM, REFUSAL_SYSTEM, docs]
    return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()


def grade(llm, q: dict, cfg: dict, agent: SupportAgent, name: str) -> dict:
    judge = settings.judge_model or None
    reply, hits = agent.answer_question(q["question"], mode=cfg["mode"], strict=cfg["strict"])
    row = {"config": name, "id": q["id"], "question": q["question"], "answer": reply.answer,
           "action": reply.action, "retrieved": [h.chunk.source for h in hits]}
    if q["answerable"]:
        row["hit"] = q["source"] in row["retrieved"]
        verdict = llm.generate_json(JUDGE_SYSTEM, f"<question>\n{q['question']}\n</question>\n"
                                    f"<expected>\n{q['expected']}\n</expected>\n<answer>\n{reply.answer}\n</answer>",
                                    model=judge)
        row["correct"], row["reason"] = bool(verdict.get("correct")), verdict.get("reason", "")
    elif reply.action == "not_found":
        row["refused"], row["reason"] = True, "guardrail"
    else:
        verdict = llm.generate_json(REFUSAL_SYSTEM, f"<question>\n{q['question']}\n</question>\n"
                                    f"<answer>\n{reply.answer}\n</answer>", model=judge)
        row["refused"], row["reason"] = bool(verdict.get("refused")), verdict.get("reason", "")
    return row


def main() -> None:
    fresh = "--fresh" in sys.argv
    questions = [json.loads(line) for line in (HERE / "questions.jsonl").read_text().splitlines() if line.strip()]
    fp = run_fingerprint()
    cache = {} if fresh or not CACHE.exists() else json.loads(CACHE.read_text())
    key = lambda name, q: hashlib.sha1(json.dumps([fp, name, q], sort_keys=True).encode()).hexdigest()  # noqa: E731

    todo = sum(key(n, q) not in cache for n in CONFIGS for q in questions)
    if todo == 0:
        print("All questions already graded with these settings; rebuilding the report.")
    elif settings.llm_provider != "fake":
        calls = 2 * todo
        print(f"{todo} of {len(CONFIGS) * len(questions)} questions to grade (the rest are saved). "
              f"About {calls} Gemini calls at {settings.gemini_rpm}/min ≈ {calls // settings.gemini_rpm + 1} min.")

    llm = get_llm()
    index = HybridIndex(llm, HERE / ".index")  # separate copy, so the running server's index isn't locked
    try:
        if todo:
            index.build(load_directory(settings.docs_dir, settings.chunk_size, settings.chunk_overlap))
            agent = SupportAgent(llm, index, Store(HERE / ".eval.db"))
        for name, cfg in CONFIGS.items():
            print(name)
            for i, q in enumerate(questions, 1):
                k = key(name, q)
                saved = k in cache
                if not saved:
                    cache[k] = grade(llm, q, cfg, agent, name)
                    CACHE.write_text(json.dumps(cache))  # save after every question
                row = cache[k]
                ok = row.get("correct", row.get("refused"))
                print(f"  [{i:>2}/{len(questions)}] {'ok  ' if ok else 'MISS'} {q['id']} {q['question'][:58]}"
                      + ("  (saved)" if saved else ""), flush=True)
    except DailyQuotaExceeded as exc:
        done = sum(key(n, q) in cache for n in CONFIGS for q in questions)
        print(f"\nStopped: {exc}\nProgress is saved ({done}/{len(CONFIGS) * len(questions)} graded). "
              "Run `python -m eval.run_eval` again after the reset to continue.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped. Progress is saved; run the same command again to continue.")
        sys.exit(1)
    finally:
        index.close()

    write_report(questions, {n: [cache[key(n, q)] for q in questions] for n in CONFIGS})


def write_report(questions: list[dict], rows_by_config: dict[str, list[dict]]) -> None:
    n_ans = sum(q["answerable"] for q in questions)
    n_un = len(questions) - n_ans
    summary, details = {}, []
    for name, rows in rows_by_config.items():
        hit = sum(r.get("hit", False) for r in rows)
        correct = sum(r.get("correct", False) for r in rows)
        refused = sum(r.get("refused", False) for r in rows)
        summary[name] = {"hit@5": f"{hit}/{n_ans}", "correct answers": f"{correct}/{n_ans}",
                         "correct refusals": f"{refused}/{n_un}",
                         "overall": f"{round(100 * (correct + refused) / len(questions))}%"}
        details += rows
        print(name, summary[name])

    (HERE / "results.json").write_text(json.dumps({"summary": summary, "details": details}, indent=1))
    judge = settings.judge_model or settings.chat_model
    model = "offline fake" if settings.llm_provider == "fake" else f"{settings.chat_model}, judged by {judge}"
    lines = [f"# Eval results ({model})", "",
             f"{len(questions)} questions: {n_ans} answerable, {n_un} not in the docs.", "",
             "| Pipeline | Retrieval hit@5 | Correct answers | Correct refusals | Overall |",
             "| --- | --- | --- | --- | --- |"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['hit@5']} | {s['correct answers']} | {s['correct refusals']} | {s['overall']} |")
    for name in rows_by_config:
        misses = [r for r in rows_by_config[name] if not (r.get("correct") or r.get("refused"))]
        lines += ["", f"## Misses: {name}", ""] + ([
            f"- **{r['id']}** {r['question']} → {r['answer'][:160]!r} ({r.get('reason', '')})" for r in misses]
            or ["None."])
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    print("Wrote eval/results.md")


if __name__ == "__main__":
    main()
