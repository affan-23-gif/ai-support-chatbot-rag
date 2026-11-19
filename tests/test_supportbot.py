"""Offline tests: they use FakeLLM, so no API key or network is needed.

Run:  pytest -q
"""
import os

os.environ["LLM_PROVIDER"] = "fake"

import shutil  # noqa: E402
import threading  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import main as main_module  # noqa: E402
from app.agent import SupportAgent  # noqa: E402
from app.config import ROOT, settings  # noqa: E402
from app.db import Store  # noqa: E402
from app.ingest import load_directory, split_text  # noqa: E402
from app.llm import DailyQuotaExceeded, FakeLLM  # noqa: E402
from app.ratelimit import RateLimiter  # noqa: E402
from app.retrieval import HybridIndex  # noqa: E402

DOCS = ROOT / "data" / "docs"
ADMIN = ("admin", "test-password")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated copy of docs, index and database for each test."""
    docs = tmp_path / "docs"
    shutil.copytree(DOCS, docs)
    for name, value in {"docs_dir": docs, "index_dir": tmp_path / "index", "db_path": tmp_path / "sb.db",
                        "admin_password": ADMIN[1], "chat_rate_limit": 1000}.items():
        monkeypatch.setattr(settings, name, value)
    return tmp_path


@pytest.fixture()
def agent(env):
    llm = FakeLLM()
    index = HybridIndex(llm, settings.index_dir)
    index.build(load_directory(settings.docs_dir, settings.chunk_size, settings.chunk_overlap))
    store = Store(settings.db_path, settings.orders_csv)
    yield SupportAgent(llm, index, store)
    index.close()
    store.close()


@pytest.fixture()
def client(env):
    with TestClient(main_module.app) as c:
        yield c


def chat(client, message, session="session-0001"):
    return client.post("/api/chat", json={"session_id": session, "message": message})


# ---------- ingestion & retrieval ----------
def test_chunks_keep_metadata():
    chunks = load_directory(DOCS, 900, 150)
    assert {"shipping.md", "gear-rental-terms.pdf", "products.csv"} <= {c.source for c in chunks}
    assert any(c.page == 2 for c in chunks if c.source.endswith(".pdf"))
    assert all(len(c.text) <= 900 + 200 for c in chunks)  # chunk + heading prefix


def test_split_text_never_loses_text():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    pieces = split_text(text, 300, 50)
    assert all(f"Paragraph {i}" in " ".join(pieces) for i in range(10))


def test_broken_file_is_skipped_not_fatal(env):
    (settings.docs_dir / "broken.pdf").write_bytes(b"not a pdf")
    sources = {c.source for c in load_directory(settings.docs_dir, 900, 150)}
    assert "broken.pdf" not in sources and "shipping.md" in sources


def test_hybrid_search_finds_pdf_page(agent):
    hits = agent.index.search("How much is a weekend tent rental?", k=3)
    assert hits[0].chunk.source == "gear-rental-terms.pdf" and hits[0].chunk.page == 1


def test_rebuild_swaps_index_and_cleans_up(agent):
    first = agent.index._snap.path
    agent.index.build(load_directory(settings.docs_dir, 900, 150))
    assert not first.exists()  # old version removed
    assert agent.index.search("standard shipping", k=1)[0].chunk.source == "shipping.md"


def test_search_keeps_working_during_rebuild(agent):
    errors = []

    def keep_searching():
        for _ in range(30):
            try:
                agent.index.search("return window", k=3)
            except Exception as exc:  # pragma: no cover - this is what we guard against
                errors.append(exc)

    t = threading.Thread(target=keep_searching)
    t.start()
    for _ in range(3):
        agent.index.build(load_directory(settings.docs_dir, 900, 150))
    t.join()
    assert not errors


def test_index_survives_restart(agent):
    agent.index.close()
    reloaded = HybridIndex(FakeLLM(), settings.index_dir)
    assert reloaded.load() and len(reloaded.chunks) == 49
    reloaded.close()


# ---------- agent behaviour ----------
def test_answer_has_citation(agent):
    reply = agent.chat("s1", "How long does standard shipping take?")
    assert reply.action == "answer" and "3-7 business days" in reply.answer
    assert reply.citations and reply.citations[0].source == "shipping.md"


def test_refuses_when_not_in_docs(agent):
    reply = agent.chat("s2", "Do you sell kayaks?")
    assert reply.action == "not_found" and not reply.citations


def test_order_lookup_needs_matching_email(agent):
    assert agent.chat("s3", "Where is my order?").action == "ask_order_details"
    ok = agent.chat("s3", "PC-10421, maya.r@example.com")
    assert ok.action == "order_status" and "Shipped" in ok.answer

    assert agent.chat("s4", "where is my order PC-10421").action == "ask_order_details"
    bad = agent.chat("s4", "someone.else@example.com")
    assert bad.action == "order_not_verified" and "Ridgeback" not in bad.answer


def test_order_lookup_locks_after_repeated_failures(agent):
    for i in range(settings.max_order_lookup_failures):
        assert agent.chat("s7", f"where is my order PC-10421 guess{i}@example.com").action == "order_not_verified"
    final = agent.chat("s7", "where is my order PC-10421 maya.r@example.com")  # even the right email now
    assert final.action == "handoff" and "Shipped" not in final.answer


def test_angry_customer_gets_one_ticket(agent):
    reply = agent.chat("s5", "This is ridiculous, my tent arrived broken")
    assert reply.action == "handoff" and reply.ticket_id
    again = agent.chat("s5", "I want to talk to a human now")
    assert again.ticket_id == reply.ticket_id  # no duplicate ticket
    assert len(agent.store.list_tickets()) == 1


def test_ticket_triggers_notifier(agent):
    sent = []
    agent.on_ticket = sent.append
    agent.chat("s6", "Can I talk to a human please")
    assert sent and sent[0]["reason"] == "customer asked for a human"


def test_session_survives_restart(agent):
    agent.chat("s8", "Where is my order?")
    fresh = SupportAgent(agent.llm, agent.index, agent.store)  # e.g. after a server restart
    assert fresh.chat("s8", "PC-10421 maya.r@example.com").action == "order_status"


def test_model_outage_gives_friendly_reply(agent, monkeypatch):
    def down(*a, **k):
        raise DailyQuotaExceeded("quota")
    monkeypatch.setattr(agent.llm, "generate_json", down)
    reply = agent.chat("s9", "How long does shipping take?")
    assert reply.action == "unavailable" and settings.support_email in reply.answer


# ---------- HTTP API ----------
def test_chat_endpoint(client):
    r = chat(client, "How long does standard shipping take?")
    assert r.status_code == 200 and r.json()["citations"][0]["source"] == "shipping.md"
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_chat_input_validation(client):
    assert chat(client, "").status_code == 422
    assert chat(client, "hi", session="bad id!").status_code == 422
    assert chat(client, "x" * 2001).status_code == 422


def test_chat_rate_limit(client):
    client.app.state.limiter = RateLimiter(limit=2)
    assert chat(client, "hi").status_code == 200
    assert chat(client, "hi").status_code == 200
    r = chat(client, "hi")
    assert r.status_code == 429 and "Retry-After" in r.headers


def test_admin_requires_password(client):
    assert client.get("/admin").status_code == 401
    assert client.get("/admin/api/tickets").status_code == 401
    assert client.get("/admin/api/tickets", auth=("admin", "wrong")).status_code == 401
    assert client.get("/admin/api/tickets", auth=ADMIN).status_code == 200
    assert client.get("/admin", auth=ADMIN).headers["X-Frame-Options"] == "DENY"


def test_admin_disabled_without_password(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_password", "")
    assert client.get("/admin/api/tickets", auth=ADMIN).status_code == 503


def test_upload_adds_knowledge(client):
    doc = b"# Kayaks\n\nWe now rent kayaks for $40 a day.\n"
    r = client.post("/admin/api/upload", auth=ADMIN, files={"file": ("kayak.md", doc, "text/markdown")})
    assert r.status_code == 200 and r.json()["chunks"] == 50
    assert chat(client, "Do you rent kayaks and how much?", "session-0002").json()["action"] == "answer"


def test_upload_rejects_bad_files_and_keeps_docs_clean(client):
    r = client.post("/admin/api/upload", auth=ADMIN, files={"file": ("fake.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400
    assert not (settings.docs_dir / "fake.pdf").exists()
    r = client.post("/admin/api/upload", auth=ADMIN, files={"file": ("run.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400
    r = client.post("/admin/api/upload", auth=ADMIN, files={"file": ("../../evil.md", b"# x\n\ny", "text/markdown")})
    assert r.status_code == 200 and (settings.docs_dir / "evil.md").exists()  # path stripped to the file name


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    big = b"# Big\n\n" + b"a" * (1024 * 1024 + 10)
    r = client.post("/admin/api/upload", auth=ADMIN, files={"file": ("big.md", big, "text/markdown")})
    assert r.status_code == 413 and not (settings.docs_dir / "big.md").exists()
