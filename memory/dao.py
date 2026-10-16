"""Agent Memory 数据访问层（DAO）。

唯一与 SQLite 打交道的层：负责落库、生成 UUID、查询。
上层领域类通过 get_dao() 获取默认 DAO 单例，不直接传 dao。
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional

from models import Query, Summary, now_iso

DEFAULT_DB_PATH = Path(__file__).parent / "memory.db"


class DAO:
    def __init__(self, db_path: str | Path = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())

    def close(self):
        self.conn.close()

    def commit(self):
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # 插入（返回数据库生成的 ID）
    # ------------------------------------------------------------------ #
    def insert_session(self) -> str:
        sid = self._new_id()
        self.conn.execute(
            "INSERT INTO session (session_id, created_at) VALUES (?, ?)",
            (sid, now_iso()),
        )
        self.commit()
        return sid

    def insert_conversation(self, session_id: str) -> str:
        cid = self._new_id()
        self.conn.execute(
            "INSERT INTO conversation (conversation_id, session_id, created_at) VALUES (?, ?, ?)",
            (cid, session_id, now_iso()),
        )
        self.commit()
        return cid

    def insert_memory(self, session_id: str) -> str:
        mid = self._new_id()
        self.conn.execute(
            "INSERT INTO memory (memory_id, session_id, created_at) VALUES (?, ?, ?)",
            (mid, session_id, now_iso()),
        )
        self.commit()
        return mid

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def insert_query(self, query: Query) -> str:
        if not query.query_id:
            query.query_id = self._new_id()
        self.conn.execute(
            "INSERT INTO query (query_id, conversation_id, summary_id, role, text, token_count, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                query.query_id,
                query.conversation_id,
                query.summary_id,
                query.role,
                query.text,
                query.token_count,
                query.timestamp,
            ),
        )
        self.commit()
        return query.query_id

    def get_queries_by_conversation(self, conversation_id: str) -> List[Query]:
        rows = self.conn.execute(
            "SELECT * FROM query WHERE conversation_id = ? ORDER BY timestamp, query_id",
            (conversation_id,),
        ).fetchall()
        return [self._row_to_query(r) for r in rows]

    def get_queries_by_summary(self, summary_id: str) -> List[Query]:
        rows = self.conn.execute(
            "SELECT * FROM query WHERE summary_id = ? ORDER BY timestamp, query_id",
            (summary_id,),
        ).fetchall()
        return [self._row_to_query(r) for r in rows]

    def get_unsummarized_queries(self, conversation_id: str) -> List[Query]:
        """未被任何摘要压缩的 query（summary_id IS NULL）。"""
        rows = self.conn.execute(
            "SELECT * FROM query WHERE conversation_id = ? AND summary_id IS NULL "
            "ORDER BY timestamp, query_id",
            (conversation_id,),
        ).fetchall()
        return [self._row_to_query(r) for r in rows]

    @staticmethod
    def _row_to_query(row: sqlite3.Row) -> Query:
        return Query(
            conversation_id=row["conversation_id"],
            role=row["role"],
            text=row["text"],
            timestamp=row["timestamp"],
            token_count=row["token_count"],
            query_id=row["query_id"],
            summary_id=row["summary_id"],
        )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    def insert_summary(self, summary: Summary) -> str:
        """落库一个 summary（含 content），并回填其关联 queries 的 summary_id。"""
        if not summary.summary_id:
            summary.summary_id = self._new_id()
        if not summary.created_at:
            summary.created_at = now_iso()
        if not summary.content:
            summary.generate_content()

        self.conn.execute(
            "INSERT INTO summary (summary_id, memory_id, conversation_id, session_id, "
            "parent_summary_id, level, content, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                summary.summary_id,
                summary.memory_id,
                summary.conversation_id,
                summary.session_id,
                summary.parent_summary_id,
                summary.level,
                summary.content,
                summary.created_at,
            ),
        )

        # 回填关联 query 的 summary_id（先落库再取 ID 后关联）
        for q in summary.queries:
            if q.query_id:
                self.conn.execute(
                    "UPDATE query SET summary_id = ? WHERE query_id = ?",
                    (summary.summary_id, q.query_id),
                )
                q.summary_id = summary.summary_id

        self.commit()
        return summary.summary_id

    def get_summaries_by_memory(self, memory_id: str) -> List[Summary]:
        rows = self.conn.execute(
            "SELECT * FROM summary WHERE memory_id = ? ORDER BY created_at, summary_id",
            (memory_id,),
        ).fetchall()
        return [self._row_to_summary(r) for r in rows]

    def get_summaries_by_session(self, session_id: str) -> List[Summary]:
        rows = self.conn.execute(
            "SELECT * FROM summary WHERE session_id = ? ORDER BY created_at, summary_id",
            (session_id,),
        ).fetchall()
        return [self._row_to_summary(r) for r in rows]

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> Summary:
        return Summary(
            queries=[],
            conversation_id=row["conversation_id"],
            session_id=row["session_id"],
            memory_id=row["memory_id"],
            level=row["level"],
            summary_id=row["summary_id"],
            content=row["content"],
            parent_summary_id=row["parent_summary_id"],
            created_at=row["created_at"],
        )


# --------------------------------------------------------------------------- #
# 默认 DAO 单例
# --------------------------------------------------------------------------- #
_default_dao: Optional[DAO] = None


def get_dao(db_path: str | Path = None) -> DAO:
    """返回默认 DAO 单例（懒创建）。"""
    global _default_dao
    if _default_dao is None:
        _default_dao = DAO(db_path)
    return _default_dao
