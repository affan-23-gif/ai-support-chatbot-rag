"""The support agent: decides what the customer wants and handles it.

Flow for every message:
  1. Route: question / order_status / human / smalltalk (+ is the customer angry?)
  2. human or angry  -> create a ticket and hand off
  3. order_status    -> collect order id + email, verify, then show status
  4. question        -> hybrid search, answer ONLY from the retrieved docs, cite sources,
                        and refuse (with a handoff offer) when the docs don't contain the answer
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

from .config import settings
from .db import Store
from .llm import LLM
from .retrieval import Hit, HybridIndex

log = logging.getLogger("supportbot.agent")

ORDER_RE = re.compile(r"\b(PC-?\d{4,6})\b", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

ROUTER_SYSTEM = """TASK: ROUTE
You classify messages sent to an online outdoor-gear store's support chat.
Return JSON: {"intent": "question" | "order_status" | "human" | "smalltalk",
              "angry": true | false,
              "standalone_question": "<the message rewritten so it makes sense without the chat history>"}
- order_status: the customer wants the status/location/tracking of THEIR specific order.
  General questions about shipping times or policies are "question".
- human: the customer asks for a person, agent or manager.
- angry: clearly upset, threatening a chargeback or review, or insulting.
- smalltalk: greetings or thanks with no question."""

ANSWER_SYSTEM = """TASK: ANSWER
You are the support assistant for Pinecrest Trail Co., an online outdoor-gear store.
Answer the customer's question using ONLY the numbered context passages.
Rules:
- If the passages do not contain the answer, set "found" to false. Never guess, never use outside knowledge.
- Do not invent prices, dates, policies or product details.
- Be friendly and brief: 1-4 sentences. Use the customer's units and wording.
- List the passage numbers you actually used in "sources".
Return JSON: {"found": true | false, "answer": "<text>", "sources": [<numbers>]}"""

LENIENT_SYSTEM = """TASK: ANSWER
You are a support assistant for an online outdoor-gear store. Answer the customer's question.
Use the numbered context passages if they help.
Return JSON: {"found": true, "answer": "<text>", "sources": [<numbers>]}"""

NOT_FOUND = ("I couldn't find that in our help documents, so I don't want to guess. "
             "Would you like me to pass this to a member of our support team?")


@dataclass
class Citation:
    n: int
    source: str
    section: str
    page: int | None
    snippet: str


@dataclass
class Reply:
    answer: str
    action: str = "answer"  # answer | not_found | ask_order_details | order_status | order_not_verified | handoff | smalltalk
    citations: list[Citation] = field(default_factory=list)
    ticket_id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Session:
    history: list[dict] = field(default_factory=list)  # [{"role": "user"|"bot", "text": ...}]
    order_id: str | None = None
    email: str | None = None
    awaiting_order_details: bool = False


class SupportAgent:
    def __init__(self, llm: LLM, index: HybridIndex, store: Store):
        self.llm = llm
        self.index = index
        self.store = store
        self.sessions: dict[str, Session] = {}

    # ---------- public ----------
    def chat(self, session_id: str, message: str) -> Reply:
        s = self.sessions.setdefault(session_id, Session())
        reply = self._handle(session_id, s, message.strip())
        s.history += [{"role": "user", "text": message}, {"role": "bot", "text": reply.answer}]
        s.history = s.history[-2 * settings.history_turns:]
        log.info("session=%s action=%s", session_id, reply.action)
        return reply

    def answer_question(self, question: str, mode: str | None = None, strict: bool = True) -> tuple[Reply, list[Hit]]:
        """Retrieve + answer. strict=False is the naive baseline used in the eval."""
        hits = self.index.search(question, k=settings.top_k, mode=mode or settings.retrieval_mode)
        # Cheap pre-check: if neither search found anything relevant, refuse without paying
        # for an LLM call. Otherwise the LLM makes the final "is it in the docs?" decision.
        best_similarity = max((h.similarity for h in hits), default=0.0)
        any_keyword_match = any(h.keyword > 0 for h in hits)
        if strict and (not hits or (best_similarity < settings.min_similarity and not any_keyword_match)):
            return Reply(NOT_FOUND, action="not_found"), hits

        context = "\n".join(
            f"[{i}] (source: {h.chunk.citation()})\n{h.chunk.text}" for i, h in enumerate(hits, start=1))
        prompt = f"<question>\n{question}\n</question>\n<context>\n{context}\n</context>"
        result = self.llm.generate_json(ANSWER_SYSTEM if strict else LENIENT_SYSTEM, prompt)

        if strict and (not result.get("found") or not result.get("answer")):
            return Reply(NOT_FOUND, action="not_found"), hits

        used = [n for n in result.get("sources", []) if isinstance(n, int) and 1 <= n <= len(hits)]
        citations = [Citation(n, hits[n - 1].chunk.source, hits[n - 1].chunk.section, hits[n - 1].chunk.page,
                              hits[n - 1].chunk.text[:220]) for n in used]
        return Reply(result.get("answer", ""), citations=citations), hits

    # ---------- internals ----------
    def _handle(self, session_id: str, s: Session, message: str) -> Reply:
        self._remember_order_details(s, message)

        # Customer is replying with the order details we asked for.
        if s.awaiting_order_details and (ORDER_RE.search(message) or EMAIL_RE.search(message)):
            return self._order_status(s)

        route = self._route(s, message)
        intent = route.get("intent", "question")

        if intent == "human" or route.get("angry"):
            reason = "customer asked for a human" if intent == "human" else "customer is upset"
            return self._handoff(session_id, s, message, reason)
        if intent == "order_status":
            return self._order_status(s)
        if intent == "smalltalk":
            return Reply("Hi! I can help with orders, shipping, returns, sizing and our gear. What do you need?",
                         action="smalltalk")

        question = route.get("standalone_question") or message
        reply, _ = self.answer_question(question)
        return reply

    def _route(self, s: Session, message: str) -> dict:
        history = "\n".join(f"{t['role']}: {t['text']}" for t in s.history[-6:])
        prompt = f"<history>\n{history}\n</history>\n<message>\n{message}\n</message>"
        try:
            return self.llm.generate_json(ROUTER_SYSTEM, prompt)
        except Exception:
            log.exception("Router failed, treating message as a question")
            return {"intent": "question", "standalone_question": message}

    @staticmethod
    def _remember_order_details(s: Session, message: str) -> None:
        if m := ORDER_RE.search(message):
            oid = m.group(1).upper()
            s.order_id = oid if "-" in oid else f"{oid[:2]}-{oid[2:]}"
        if m := EMAIL_RE.search(message):
            s.email = m.group(0).lower()

    def _order_status(self, s: Session) -> Reply:
        missing = [name for name, v in (("order number (like PC-10421)", s.order_id),
                                        ("email address used for the order", s.email)) if not v]
        if missing:
            s.awaiting_order_details = True
            return Reply(f"I can check that. Please send your {' and '.join(missing)}.",
                         action="ask_order_details")

        s.awaiting_order_details = False
        order = self.store.find_order(s.order_id, s.email)
        if not order:
            # Same message whether the id or the email is wrong: don't leak which orders exist.
            s.order_id = s.email = None
            return Reply("I couldn't verify an order with that number and email. Please check both and try "
                         "again, or ask me to connect you with our team.", action="order_not_verified")

        text = f"Order {order['order_id']} ({order['items']}) is **{order['status']}**."
        if order["tracking"]:
            text += f" Carrier: {order['carrier']}, tracking number {order['tracking']}."
        if order["eta"]:
            text += f" Estimated delivery: {order['eta']}."
        return Reply(text, action="order_status")

    def _handoff(self, session_id: str, s: Session, message: str, reason: str) -> Reply:
        ticket_id = self.store.create_ticket(session_id, reason, message, s.history + [{"role": "user", "text": message}])
        return Reply(f"I'm sorry about this. I've passed your conversation to our support team (ticket #{ticket_id}). "
                     "A person will reply by email within 1 business day.", action="handoff", ticket_id=ticket_id)
