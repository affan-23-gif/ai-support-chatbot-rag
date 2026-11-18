"""SQLite storage for the mock order system and human-handoff tickets."""
from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class Store:
    def __init__(self, db_path: Path, orders_csv: Path | None = None):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY, email TEXT, status TEXT, items TEXT,
                carrier TEXT, tracking TEXT, eta TEXT);
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, session_id TEXT,
                reason TEXT, last_message TEXT, transcript TEXT, status TEXT DEFAULT 'open');
        """)
        if orders_csv and orders_csv.exists():
            self.load_orders(orders_csv)

    def load_orders(self, path: Path) -> None:
        with path.open(newline="", encoding="utf-8") as f:
            rows = [(r["order_id"].upper(), r["email"].lower(), r["status"], r["items"],
                     r["carrier"], r["tracking"], r["eta"]) for r in csv.DictReader(f)]
        self.conn.executemany("INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?)", rows)
        self.conn.commit()

    def find_order(self, order_id: str, email: str) -> dict | None:
        """Return the order only if BOTH the id and the email match (simple verification)."""
        row = self.conn.execute(
            "SELECT * FROM orders WHERE order_id = ? AND email = ?",
            (order_id.strip().upper(), email.strip().lower())).fetchone()
        return dict(row) if row else None

    def create_ticket(self, session_id: str, reason: str, last_message: str, transcript: list[dict]) -> int:
        cur = self.conn.execute(
            "INSERT INTO tickets (created_at, session_id, reason, last_message, transcript) VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), session_id, reason, last_message,
             json.dumps(transcript)))
        self.conn.commit()
        return cur.lastrowid

    def list_tickets(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM tickets ORDER BY id DESC").fetchall()
        return [{**dict(r), "transcript": json.loads(r["transcript"])} for r in rows]
