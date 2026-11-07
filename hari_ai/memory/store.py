"""Explicitly saved local memories; never used for training automatically."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import List, Dict, Any, Optional


def default_memory_path() -> Path:
    return Path(os.environ.get('HARIAI_DATA_DIR', Path.home() / '.hari_ai')) / 'memory.sqlite3'


class MemoryStore:
    """Opt-in local SQLite storage. Contents are plaintext and remain on this device."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else default_memory_path()

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.execute('CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, content TEXT NOT NULL, '
                     'sha256 TEXT NOT NULL UNIQUE, created_at REAL NOT NULL)')
        return conn

    def add(self, text: str) -> int:
        text = text.strip()
        if not text or len(text) > 10000:
            raise ValueError('Memory must contain between 1 and 10000 characters')
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        with closing(self._connect()) as conn, conn:
            conn.execute('INSERT OR IGNORE INTO memories (content, sha256, created_at) VALUES (?, ?, ?)',
                         (text, digest, time.time()))
            return int(conn.execute('SELECT id FROM memories WHERE sha256 = ?', (digest,)).fetchone()[0])

    def list(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with closing(self._connect()) as conn, conn:
            return [{'id': row[0], 'content': row[1], 'created_at': row[2]}
                    for row in conn.execute('SELECT id, content, created_at FROM memories ORDER BY id')]

    def delete(self, memory_id: int) -> bool:
        if not self.path.exists():
            return False
        with closing(self._connect()) as conn, conn:
            return conn.execute('DELETE FROM memories WHERE id = ?', (memory_id,)).rowcount > 0

