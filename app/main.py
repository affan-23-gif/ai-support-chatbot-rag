"""FastAPI app: chat API, admin API (password protected), and the web pages."""
from __future__ import annotations

import logging
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import SupportAgent
from .config import ROOT, settings
from .db import Store
from .ingest import SUPPORTED, UnreadableDocument, load_directory, load_file
from .llm import get_llm
from .notify import make_ticket_notifier
from .ratelimit import RateLimiter
from .retrieval import HybridIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("supportbot")
STATIC = ROOT / "static"
_reindex_lock = threading.Lock()  # one rebuild at a time
_basic = HTTPBasic(auto_error=False)


def reindex(index: HybridIndex) -> int:
    if not _reindex_lock.acquire(blocking=False):
        raise HTTPException(409, "A re-index is already running. Try again in a moment.")
    try:
        chunks = load_directory(settings.docs_dir, settings.chunk_size, settings.chunk_overlap)
        index.build(chunks)
        return len(chunks)
    finally:
        _reindex_lock.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = get_llm()
    index = HybridIndex(llm, settings.index_dir)
    if not index.load():
        log.info("No index found, building it from %s", settings.docs_dir)
        reindex(index)
    store = Store(settings.db_path, settings.orders_csv)
    purged = store.purge_sessions(settings.session_ttl_hours)
    if not settings.admin_password:
        log.warning("ADMIN_PASSWORD is not set: /admin and the admin API are switched off.")
    app.state.index = index
    app.state.agent = SupportAgent(llm, index, store, on_ticket=make_ticket_notifier(settings.ticket_webhook_url))
    app.state.limiter = RateLimiter(settings.chat_rate_limit)
    log.info("Ready: %d chunks, %d expired sessions removed", len(index.chunks), purged)
    yield
    index.close()
    store.close()


app = FastAPI(title="SupportBot", lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def log_and_harden(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("req=%s %s %s crashed", request_id, request.method, request.url.path)
        response = JSONResponse({"detail": "Internal error"}, status_code=500)
    ms = (time.perf_counter() - start) * 1000
    if request.url.path.startswith(("/api/", "/admin/api/")):
        log.info("req=%s %s %s %d %.0fms", request_id, request.method, request.url.path, response.status_code, ms)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/admin"):
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
    return response


def client_ip(request: Request) -> str:
    if settings.trust_proxy and (fwd := request.headers.get("x-forwarded-for")):
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_admin(creds: HTTPBasicCredentials | None = Depends(_basic)) -> None:
    if not settings.admin_password:
        raise HTTPException(503, "Admin is switched off. Set ADMIN_PASSWORD in .env to enable it.")
    ok = creds is not None and secrets.compare_digest(creds.username.encode(), b"admin") \
        and secrets.compare_digest(creds.password.encode(), settings.admin_password.encode())
    if not ok:
        raise HTTPException(401, "Admin login required.", headers={"WWW-Authenticate": 'Basic realm="SupportBot admin"'})


# ---------- public ----------
class ChatIn(BaseModel):
    session_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=2000)


@app.get("/")
def chat_page():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": len(app.state.index.chunks)}


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    allowed, retry_after = app.state.limiter.allow(client_ip(request))
    if not allowed:
        raise HTTPException(429, "You're sending messages too quickly. Please wait a moment.",
                            headers={"Retry-After": str(retry_after)})
    message = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", body.message).strip()
    if not message:
        raise HTTPException(422, "Message is empty.")
    return app.state.agent.chat(body.session_id, message).to_dict()


# ---------- admin (HTTP Basic auth, user "admin") ----------
@app.get("/admin", dependencies=[Depends(require_admin)])
def admin_page():
    return FileResponse(STATIC / "admin.html")


@app.post("/admin/api/ingest", dependencies=[Depends(require_admin)])
def ingest():
    return {"chunks": reindex(app.state.index)}


@app.post("/admin/api/upload", dependencies=[Depends(require_admin)])
def upload(file: UploadFile = File(...)):
    name = re.sub(r"[^\w.\-]", "_", Path(file.filename or "").name).lstrip(".")
    if not name or Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(400, f"Only {', '.join(sorted(SUPPORTED))} files are supported.")

    limit = settings.max_upload_mb * 1024 * 1024
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / name
        size = 0
        with tmp_path.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")
                out.write(chunk)
        try:  # parse it BEFORE it touches the live docs folder
            n_file = len(load_file(tmp_path, settings.chunk_size, settings.chunk_overlap))
        except (UnreadableDocument, UnicodeDecodeError) as exc:
            raise HTTPException(400, str(exc) if isinstance(exc, UnreadableDocument)
                                else f"{name} is not valid UTF-8 text.")

        target = settings.docs_dir / name
        backup = target.read_bytes() if target.exists() else None
        shutil.copyfile(tmp_path, target)
    try:
        total = reindex(app.state.index)
    except Exception:
        # Put the docs folder back the way it was; the live index was never replaced.
        if backup is None:
            target.unlink(missing_ok=True)
        else:
            target.write_bytes(backup)
        raise
    return {"saved": name, "chunks_in_file": n_file, "chunks": total}


@app.get("/admin/api/documents", dependencies=[Depends(require_admin)])
def documents():
    return sorted(p.name for p in settings.docs_dir.iterdir() if p.suffix.lower() in SUPPORTED)


@app.get("/admin/api/tickets", dependencies=[Depends(require_admin)])
def tickets():
    return app.state.agent.store.list_tickets()
