"""FastAPI app: chat API, document upload/re-index, tickets, and the web pages."""
from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import SupportAgent
from .config import ROOT, settings
from .db import Store
from .ingest import load_directory
from .llm import get_llm
from .retrieval import HybridIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("supportbot")
STATIC = ROOT / "static"
ALLOWED = {".md", ".txt", ".pdf", ".csv"}


def reindex(index: HybridIndex) -> int:
    chunks = load_directory(settings.docs_dir, settings.chunk_size, settings.chunk_overlap)
    index.build(chunks)
    return len(chunks)


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = get_llm()
    index = HybridIndex(llm, settings.index_dir)
    if not index.load():
        log.info("No index found, building it from %s", settings.docs_dir)
        reindex(index)
    store = Store(settings.db_path, settings.orders_csv)
    app.state.index = index
    app.state.agent = SupportAgent(llm, index, store)
    yield


app = FastAPI(title="SupportBot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=2000)


@app.get("/")
def chat_page():
    return FileResponse(STATIC / "index.html")


@app.get("/admin")
def admin_page():
    return FileResponse(STATIC / "admin.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": len(app.state.index.chunks), "provider": settings.llm_provider}


@app.post("/api/chat")
def chat(body: ChatIn):
    try:
        return app.state.agent.chat(body.session_id, body.message).to_dict()
    except Exception:
        log.exception("Chat failed")
        raise HTTPException(503, "The assistant is having trouble right now. Please try again in a moment.")


@app.post("/api/ingest")
def ingest():
    return {"chunks": reindex(app.state.index)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = re.sub(r"[^\w.\-]", "_", Path(file.filename or "").name)
    if Path(name).suffix.lower() not in ALLOWED:
        raise HTTPException(400, f"Only {', '.join(sorted(ALLOWED))} files are supported.")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, "File is larger than 10 MB.")
    (settings.docs_dir / name).write_bytes(data)
    return {"saved": name, "chunks": reindex(app.state.index)}


@app.get("/api/documents")
def documents():
    return sorted(p.name for p in settings.docs_dir.iterdir() if p.suffix.lower() in ALLOWED)


@app.get("/api/tickets")
def tickets():
    return app.state.agent.store.list_tickets()
