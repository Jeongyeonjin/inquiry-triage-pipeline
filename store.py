# -*- coding: utf-8 -*-
"""문의 저장소(SQLite). 문의 상태와 단계별 실행 로그를 함께 보관한다."""
import sqlite3, time
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS inquiries(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  external_id TEXT UNIQUE,            -- 보내는 쪽이 붙인 ID. 같은 ID가 다시 오면 중복으로 처리한다
  received_at REAL NOT NULL,
  text TEXT,
  status TEXT NOT NULL DEFAULT 'received',  -- received / processed / needs_review / reviewed / failed
  category TEXT, confidence REAL, summary TEXT, reason TEXT,
  tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0, cost REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS logs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  inquiry_id INTEGER, at REAL NOT NULL, step TEXT, level TEXT, message TEXT
);
"""
UPDATABLE = {"status", "category", "confidence", "summary", "reason", "tokens_in", "tokens_out", "cost"}


class Store:
    def __init__(self, path="inquiries.db"):
        self.path = path
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def add_inquiry(self, external_id, text, now=None):
        """(id, 새로 만들었는지) 를 돌려준다. 같은 external_id면 기존 문의를 돌려주고 False."""
        now = time.time() if now is None else now
        with self._conn() as c:
            try:
                cur = c.execute("INSERT INTO inquiries(external_id, received_at, text) VALUES (?,?,?)",
                                (external_id, now, text))
                return cur.lastrowid, True
            except sqlite3.IntegrityError:
                row = c.execute("SELECT id FROM inquiries WHERE external_id=?", (external_id,)).fetchone()
                return row["id"], False

    def get(self, inquiry_id):
        with self._conn() as c:
            row = c.execute("SELECT * FROM inquiries WHERE id=?", (inquiry_id,)).fetchone()
        return dict(row) if row else None

    def update(self, inquiry_id, **fields):
        bad = set(fields) - UPDATABLE
        if bad:
            raise ValueError(f"수정할 수 없는 필드: {bad}")
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE inquiries SET {sets} WHERE id=?", (*fields.values(), inquiry_id))

    def log(self, inquiry_id, step, level, message, now=None):
        now = time.time() if now is None else now
        with self._conn() as c:
            c.execute("INSERT INTO logs(inquiry_id, at, step, level, message) VALUES (?,?,?,?,?)",
                      (inquiry_id, now, step, level, message))

    def logs_for(self, inquiry_id):
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM logs WHERE inquiry_id=? ORDER BY id", (inquiry_id,))]

    def list_by_status(self, status):
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM inquiries WHERE status=? ORDER BY id", (status,))]

    def between(self, since, until):
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM inquiries WHERE received_at>=? AND received_at<? ORDER BY id", (since, until))]
