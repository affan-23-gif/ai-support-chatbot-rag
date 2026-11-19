"""Load documents and split them into chunks that keep their source metadata.

Supported: .md / .txt (split by headings), .pdf (per page), .csv (one chunk per row).
Every chunk remembers its file, page and section so answers can cite it.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path


log = logging.getLogger("supportbot.ingest")


class UnreadableDocument(ValueError):
    """The file can't be parsed or has no text (e.g. a scanned PDF without OCR)."""


@dataclass
class Chunk:
    id: str
    text: str
    source: str              # file name
    section: str = ""        # heading the text sits under
    page: int | None = None  # PDF page number, 1-based

    def citation(self) -> str:
        where = f"p. {self.page}" if self.page else self.section
        return f"{self.source}" + (f" › {where}" if where else "")

    def to_dict(self) -> dict:
        return asdict(self)


def _chunk_id(source: str, n: int, text: str) -> str:
    return hashlib.sha1(f"{source}:{n}:{text[:50]}".encode()).hexdigest()[:16]


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph boundaries; only cut inside a paragraph when it is too long."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= size:
            current = f"{current}\n\n{p}" if current else p
            continue
        if current:
            pieces.append(current)
        while len(p) > size:  # very long paragraph: hard split with overlap
            pieces.append(p[:size])
            p = p[size - overlap:]
        current = p
    if current:
        pieces.append(current)
    return pieces


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading path, body) pairs, e.g. ('Returns › Refund timing', '...')."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    body: list[str] = []

    def flush():
        if "".join(body).strip():
            sections.append((" › ".join(path), "\n".join(body).strip()))

    for line in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            body = []
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
        else:
            body.append(line)
    flush()
    return sections


def load_file(path: Path, size: int, overlap: int) -> list[Chunk]:
    name = path.name
    suffix = path.suffix.lower()
    chunks: list[Chunk] = []

    if suffix in {".md", ".txt"}:
        n = 0
        for section, body in _markdown_sections(path.read_text(encoding="utf-8")):
            for piece in split_text(body, size, overlap):
                # Put the heading in the text too: it helps both keyword and vector search.
                text = f"{section}\n{piece}" if section else piece
                chunks.append(Chunk(_chunk_id(name, n, text), text, name, section))
                n += 1

    elif suffix == ".pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        n = 0
        try:
            pages = PdfReader(str(path)).pages
            for page_no, page in enumerate(pages, start=1):
                for piece in split_text(page.extract_text() or "", size, overlap):
                    chunks.append(Chunk(_chunk_id(name, n, piece), piece, name, page=page_no))
                    n += 1
        except (PdfReadError, ValueError, KeyError) as exc:
            raise UnreadableDocument(f"{name} is not a readable PDF ({exc}).") from exc

    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            for n, row in enumerate(csv.DictReader(f)):
                text = "; ".join(f"{k}: {v}" for k, v in row.items() if v)
                label = row.get("name") or row.get("sku") or f"row {n + 1}"
                chunks.append(Chunk(_chunk_id(name, n, text), text, name, section=label))
    if not chunks:
        raise UnreadableDocument(f"No text found in {name}. Scanned PDFs need OCR first.")
    return chunks


SUPPORTED = {".md", ".txt", ".pdf", ".csv"}


def load_directory(docs_dir: Path, size: int, overlap: int) -> list[Chunk]:
    """Load every supported file. A broken file is logged and skipped, not fatal."""
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED:
            try:
                chunks.extend(load_file(path, size, overlap))
            except (UnreadableDocument, UnicodeDecodeError, csv.Error) as exc:
                log.warning("Skipping %s: %s", path.name, exc)
    return chunks
