"""Offline tests: they use FakeLLM, so no API key or network is needed.

Run:  pytest -q
"""
import os

os.environ["LLM_PROVIDER"] = "fake"

import pytest  # noqa: E402

from app.agent import SupportAgent  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import Store  # noqa: E402
from app.ingest import load_directory, split_text  # noqa: E402
from app.llm import FakeLLM  # noqa: E402
from app.retrieval import HybridIndex  # noqa: E402


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sb")
    llm = FakeLLM()
    index = HybridIndex(llm, tmp / "index")
    index.build(load_directory(settings.docs_dir, settings.chunk_size, settings.chunk_overlap))
    return SupportAgent(llm, index, Store(tmp / "test.db", settings.orders_csv))


def test_chunks_keep_metadata():
    chunks = load_directory(settings.docs_dir, 900, 150)
    sources = {c.source for c in chunks}
    assert {"shipping.md", "gear-rental-terms.pdf", "products.csv"} <= sources
    assert any(c.page == 2 for c in chunks if c.source.endswith(".pdf"))
    assert all(len(c.text) <= 900 + 200 for c in chunks)  # chunk + heading prefix


def test_split_text_never_loses_text():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    pieces = split_text(text, 300, 50)
    assert all(f"Paragraph {i}" in " ".join(pieces) for i in range(10))


def test_hybrid_search_finds_pdf_page(agent):
    hits = agent.index.search("How much is a weekend tent rental?", k=3)
    assert hits[0].chunk.source == "gear-rental-terms.pdf" and hits[0].chunk.page == 1


def test_answer_has_citation(agent):
    reply = agent.chat("s1", "How long does standard shipping take?")
    assert reply.action == "answer"
    assert "3-7 business days" in reply.answer
    assert reply.citations and reply.citations[0].source == "shipping.md"


def test_refuses_when_not_in_docs(agent):
    reply = agent.chat("s2", "Do you sell kayaks?")
    assert reply.action == "not_found"
    assert not reply.citations


def test_order_lookup_needs_matching_email(agent):
    assert agent.chat("s3", "Where is my order?").action == "ask_order_details"
    ok = agent.chat("s3", "PC-10421, maya.r@example.com")
    assert ok.action == "order_status" and "Shipped" in ok.answer

    assert agent.chat("s4", "where is my order PC-10421").action == "ask_order_details"
    bad = agent.chat("s4", "someone.else@example.com")
    assert bad.action == "order_not_verified"
    assert "Ridgeback" not in bad.answer  # no order details leaked


def test_angry_customer_gets_ticket(agent):
    reply = agent.chat("s5", "This is ridiculous, my tent arrived broken")
    assert reply.action == "handoff" and reply.ticket_id
    ticket = agent.store.list_tickets()[0]
    assert ticket["session_id"] == "s5" and ticket["transcript"][-1]["text"].startswith("This is ridiculous")


def test_human_request_gets_ticket(agent):
    assert agent.chat("s6", "Can I talk to a human please").action == "handoff"
