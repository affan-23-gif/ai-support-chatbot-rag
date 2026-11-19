"""LLM layer.

Everything that talks to a model goes through two methods:
  - embed(texts, task)  -> list of vectors
  - generate_json(system, prompt) -> dict

GeminiLLM calls the real Gemini API. FakeLLM is a small offline stand-in so the
whole app, the tests and the eval script can run without an API key or network.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
from typing import Protocol

from .config import settings

log = logging.getLogger("supportbot.llm")


class LLM(Protocol):
    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]: ...
    def generate_json(self, system: str, prompt: str, model: str | None = None) -> dict: ...


class DailyQuotaExceeded(RuntimeError):
    """The free tier's per-day request limit is used up. Retrying today won't help."""


def _parse_json(text: str) -> dict:
    """Models sometimes wrap JSON in ``` fences. Strip them and parse."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


class GeminiLLM:
    def __init__(self, api_key: str, chat_model: str, embedding_model: str, dim: int, rpm: int = 12):
        from google import genai  # imported here so FakeLLM works without the SDK

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
        self.client = genai.Client(api_key=api_key)
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.dim = dim
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next_slot: dict[str, float] = {}  # quotas are per model, so pace each model separately
        self._lock = threading.Lock()

    def _throttle(self, model: str) -> None:
        """Space out generate calls so we stay under the requests-per-minute quota."""
        with self._lock:
            slot = self._next_slot.get(model, 0.0)
            wait = slot - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._next_slot[model] = max(slot, time.monotonic()) + self.min_interval

    def _retry(self, fn, attempts: int = 5):
        """Retry transient errors.

        - Daily quota used up: stop at once with a clear message (waiting won't help today).
        - Per-minute quota (429): wait as long as the API asks, up to 65 s.
        - Other errors (503 overloaded, timeouts): exponential backoff.
        """
        for i in range(attempts):
            try:
                return fn()
            except Exception as exc:  # the SDK raises several error types
                msg = str(exc)
                if "PerDay" in msg or "per day" in msg:
                    h = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?", msg)
                    reset = (f" It resets in about {h.group(1) or 0}h {h.group(2) or 0}m."
                             if h and (h.group(1) or h.group(2)) else "")
                    raise DailyQuotaExceeded(
                        "Gemini free-tier DAILY request limit reached for this model." + reset +
                        " Wait for the reset, switch CHAT_MODEL / JUDGE_MODEL in .env, or enable billing.") from exc
                if i == attempts - 1:
                    raise
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    m = re.search(r"retry in ([\d.]+)s|'retryDelay': '(\d+)s'", msg)
                    wait = min(float(m.group(1) or m.group(2)) + 1, 65) if m else 30
                    log.warning("Gemini rate limit hit, waiting %.0fs (attempt %d/%d)", wait, i + 1, attempts)
                else:
                    wait = 2 ** i
                    log.warning("Gemini call failed (%s), retrying in %ss", msg[:200], wait)
                time.sleep(wait)

    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        from google.genai import types

        task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 50):  # batch to stay under request limits
            batch = texts[start:start + 50]
            resp = self._retry(lambda: self.client.models.embed_content(
                model=self.embedding_model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
            ))
            vectors.extend(e.values for e in resp.embeddings)
        return [_normalize(v) for v in vectors]

    def generate_json(self, system: str, prompt: str, model: str | None = None) -> dict:
        from google.genai import types

        model = model or self.chat_model
        self._throttle(model)
        resp = self._retry(lambda: self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                # We don't pass tools, so turn off the SDK's auto tool-calling (removes the "AFC" log line).
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        ))
        return _parse_json(resp.text or "{}")


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the is are was were do does did i you we our your my me to of in on for with and or "
            "can could how what when where which who why it this that be at by from as if not".split())


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class FakeLLM:
    """Offline stand-in. Embeddings are hashed bags of words; generation is rule based.

    It is good enough to exercise every code path. Real answer quality needs Gemini.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in tokenize(t):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            out.append(_normalize(v))
        return out

    def generate_json(self, system: str, prompt: str, model: str | None = None) -> dict:
        if "TASK: ROUTE" in system:
            return self._route(prompt)
        if "TASK: ANSWER" in system:
            return self._answer(prompt)
        if "TASK: REFUSAL_JUDGE" in system:
            answer = self._section(prompt, "answer").lower()
            return {"refused": bool(re.search(r"don't know|couldn't find|not sure|no information", answer))}
        if "TASK: JUDGE" in system:
            return self._judge(prompt)
        return {}

    @staticmethod
    def _section(prompt: str, name: str) -> str:
        m = re.search(rf"<{name}>\n?(.*?)\n?</{name}>", prompt, re.S)
        return m.group(1).strip() if m else ""

    def _route(self, prompt: str) -> dict:
        msg = self._section(prompt, "message").lower()
        intent = "question"
        if re.search(r"\b(human|agent|person|representative|manager)\b", msg):
            intent = "human"
        elif re.search(r"\b(order|package|parcel|tracking|delivery status|where is my)\b", msg) and \
                not re.search(r"\b(how|policy|cost|long|can i)\b", msg):
            intent = "order_status"
        elif re.fullmatch(r"(hi|hello|hey|thanks|thank you)[!. ]*", msg):
            intent = "smalltalk"
        angry = bool(re.search(r"\b(angry|furious|ridiculous|terrible|worst|scam|unacceptable)\b", msg))
        return {"intent": intent, "angry": angry, "standalone_question": self._section(prompt, "message")}

    def _answer(self, prompt: str) -> dict:
        question = set(tokenize(self._section(prompt, "question")))
        best, best_overlap = None, 0
        for m in re.finditer(r"\[(\d+)\] \(source: [^)]*\)\n(.*?)(?=\n\[\d+\] \(source:|\Z)",
                             self._section(prompt, "context"), re.S):
            overlap = len(question & set(tokenize(m.group(2))))
            if overlap > best_overlap:
                best, best_overlap = m, overlap
        if not best or best_overlap < 2:
            return {"found": False, "answer": "", "sources": []}
        sentences = re.split(r"(?<=[.!?])\s+", best.group(2).strip())
        scored = sorted(sentences, key=lambda s: -len(question & set(tokenize(s))))
        return {"found": True, "answer": " ".join(scored[:2]), "sources": [int(best.group(1))]}

    def _judge(self, prompt: str) -> dict:
        answer = self._section(prompt, "answer").lower()
        keys = [k.strip().lower() for k in self._section(prompt, "expected").split("|") if k.strip()]
        hit = sum(1 for k in keys if k in answer)
        return {"correct": bool(keys) and hit >= max(1, math.ceil(len(keys) / 2)), "reason": f"{hit}/{len(keys)} key facts"}


def get_llm() -> LLM:
    if settings.llm_provider == "fake":
        return FakeLLM()
    return GeminiLLM(settings.gemini_api_key, settings.chat_model, settings.embedding_model,
                     settings.embedding_dim, settings.gemini_rpm)
