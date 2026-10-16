#!/usr/bin/env python3
"""创建 Agent Memory 的 SQLite 数据库（memory.db）。"""
import sqlite3
from pathlib import Path

BASE = Path(__file__).parent
DB_PATH = BASE / "memory.db"
SCHEMA_PATH = BASE / "schema.sql"


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()

    print(f"数据库已创建: {DB_PATH}")
    print("表:", [t[0] for t in tables])

    # 打印每张表的列信息确认
    for (name,) in tables:
        cols = conn.execute(f"PRAGMA table_info({name})").fetchall()
        col_names = [c[1] for c in cols]
        print(f"  {name}: {', '.join(col_names)}")

    conn.close()


if __name__ == "__main__":
    main()
