"""A small ticket system on SQLite, seeded from data/tickets.json."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from . import DATA_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    description TEXT NOT NULL,
    requester   TEXT NOT NULL,
    category    TEXT NOT NULL,
    priority    TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL
)
"""

_COLUMNS = ("id", "title", "description", "requester", "category", "priority", "status", "created_at")
_UPDATABLE = {"title", "description", "category", "priority", "status"}


class TicketStore:
    """lookup_ticket, create_ticket and update_ticket over one SQLite database."""

    def __init__(self, path: str = ":memory:", seed: bool = True):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(_SCHEMA)
        if seed and self.conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0:
            self._seed()
        self.conn.commit()

    def _seed(self) -> None:
        tickets = json.loads((DATA_DIR / "tickets.json").read_text(encoding="utf-8"))
        self.conn.executemany(
            f"INSERT INTO tickets ({', '.join(_COLUMNS)}) "
            f"VALUES ({', '.join(':' + column for column in _COLUMNS)})",
            tickets,
        )

    def lookup_ticket(self, ticket_id: int) -> dict | None:
        """Return the ticket as a dict, or None if it does not exist."""
        row = self.conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return dict(row) if row else None

    def create_ticket(
        self,
        title: str,
        description: str,
        requester: str,
        category: str = "other",
        priority: str = "P3",
    ) -> dict:
        """Open a new ticket and return it, including its new id."""
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cursor = self.conn.execute(
            "INSERT INTO tickets (title, description, requester, category, priority, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'open', ?)",
            (title, description, requester, category, priority, created_at),
        )
        self.conn.commit()
        return self.lookup_ticket(cursor.lastrowid)

    def update_ticket(self, ticket_id: int, **fields) -> dict:
        """Change fields on an existing ticket and return the new version."""
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"Cannot update {sorted(unknown)}. Allowed: {sorted(_UPDATABLE)}")
        if self.lookup_ticket(ticket_id) is None:
            raise ValueError(f"Ticket {ticket_id} does not exist")
        if fields:
            assignments = ", ".join(f"{name} = ?" for name in fields)
            self.conn.execute(
                f"UPDATE tickets SET {assignments} WHERE id = ?",
                (*fields.values(), ticket_id),
            )
            self.conn.commit()
        return self.lookup_ticket(ticket_id)
