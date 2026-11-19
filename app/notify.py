"""Post new handoff tickets to a webhook (Slack incoming webhook format works as is)."""
from __future__ import annotations

import json
import logging
import threading
import urllib.request

log = logging.getLogger("supportbot.notify")


def make_ticket_notifier(url: str):
    if not url:
        return None

    def send(ticket: dict) -> None:
        text = (f"New support ticket #{ticket['id']} ({ticket['reason']})\n"
                f"Last message: {ticket['last_message'][:300]}")
        body = json.dumps({"text": text}).encode()

        def post() -> None:
            try:
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5).read()
            except Exception as exc:  # a notification must never break the chat
                log.warning("Ticket webhook failed: %s", exc)

        threading.Thread(target=post, daemon=True).start()

    return send
