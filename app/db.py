"""SQLite storage: mock orders, human-handoff tickets and chat sessions.

One connection is shared across FastAPI's worker threads, so every access goes
through a lock. SQLite is fine for one server process; see the README for scaling.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Store:
    def __init__(self, db_path: Path, orders_csv: Path | None = None):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY, email TEXT, status TEXT, items TEXT,
                    carrier TEXT, tracking TEXT, eta TEXT);
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, session_id TEXT,
                    reason TEXT, last_message TEXT, transcript TEXT, status TEXT DEFAULT 'open');
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TEXT NOT NULL);
            """)
        if orders_csv and orders_csv.exists():
            self.load_orders(orders_csv)

    # ---------- orders ----------
    def load_orders(self, path: Path) -> None:
        with path.open(newline="", encoding="utf-8") as f:
            rows = [(r["order_id"].upper(), r["email"].lower(), r["status"], r["items"],
                     r["carrier"], r["tracking"], r["eta"]) for r in csv.DictReader(f)]
        with self._lock:
            self.conn.executemany("INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?)", rows)
            self.conn.commit()

    def find_order(self, order_id: str, email: str) -> dict | None:
        """Return the order only if BOTH the id and the email match (simple verification)."""
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM orders WHERE order_id = ? AND email = ?",
                (order_id.strip().upper(), email.strip().lower())).fetchone()
        return dict(row) if row else None

    # ---------- tickets ----------
    def create_ticket(self, session_id: str, reason: str, last_message: str, transcript: list[dict]) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO tickets (created_at, session_id, reason, last_message, transcript) VALUES (?,?,?,?,?)",
                (_now().isoformat(timespec="seconds"), session_id, reason, last_message, json.dumps(transcript)))
            self.conn.commit()
            return cur.lastrowid

    def list_tickets(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM tickets ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "transcript": json.loads(r["transcript"])} for r in rows]

    # ---------- sessions ----------
    def load_session(self, session_id: str, ttl_hours: int) -> dict | None:
        cutoff = (_now() - timedelta(hours=ttl_hours)).isoformat()
        with self._lock:
            row = self.conn.execute("SELECT data FROM sessions WHERE id = ? AND updated_at >= ?",
                                    (session_id, cutoff)).fetchone()
        return json.loads(row["data"]) if row else None

    def save_session(self, session_id: str, data: dict) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?)",
                              (session_id, json.dumps(data), _now().isoformat()))
            self.conn.commit()

    def purge_sessions(self, ttl_hours: int) -> int:
        cutoff = (_now() - timedelta(hours=ttl_hours)).isoformat()
        with self._lock:
            n = self.conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,)).rowcount
            self.conn.commit()
        return n

    def close(self) -> None:
        with self._lock:
            self.conn.close()
