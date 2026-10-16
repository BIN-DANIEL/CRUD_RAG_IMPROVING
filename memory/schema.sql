-- =============================================
-- Agent Memory Schema (SQLite)
-- 主键统一使用 UUID (TEXT)，表名与领域类同名（单数）
-- =============================================

PRAGMA foreign_keys = ON;

-- 1. session：会话容器
CREATE TABLE IF NOT EXISTS session (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 2. conversation：对话（1:1 属于 session）
CREATE TABLE IF NOT EXISTS conversation (
    conversation_id TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL UNIQUE,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES session(session_id)
);

-- 3. memory：记忆（1:1 属于 session）
CREATE TABLE IF NOT EXISTS memory (
    memory_id   TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES session(session_id)
);

-- 4. summary：摘要（支持 parent 自引用做二级摘要）
CREATE TABLE IF NOT EXISTS summary (
    summary_id        TEXT PRIMARY KEY,
    memory_id         TEXT NOT NULL,
    conversation_id   TEXT NOT NULL,
    session_id        TEXT NOT NULL,
    parent_summary_id TEXT,
    level             INTEGER NOT NULL DEFAULT 1,
    content           TEXT NOT NULL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (memory_id)         REFERENCES memory(memory_id),
    FOREIGN KEY (conversation_id)   REFERENCES conversation(conversation_id),
    FOREIGN KEY (session_id)        REFERENCES session(session_id),
    FOREIGN KEY (parent_summary_id) REFERENCES summary(summary_id)
);

-- 5. query：原始对话记录
CREATE TABLE IF NOT EXISTS query (
    query_id        TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    summary_id      TEXT,
    role            TEXT NOT NULL CHECK (role IN ('model', 'user')),
    text            TEXT NOT NULL,
    token_count     INTEGER,
    timestamp       TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversation(conversation_id),
    FOREIGN KEY (summary_id)     REFERENCES summary(summary_id)
);

-- 索引：加速按外键查询
CREATE INDEX IF NOT EXISTS idx_query_conversation ON query(conversation_id);
CREATE INDEX IF NOT EXISTS idx_query_summary      ON query(summary_id);
CREATE INDEX IF NOT EXISTS idx_summary_memory     ON summary(memory_id);
CREATE INDEX IF NOT EXISTS idx_summary_parent     ON summary(parent_summary_id);

