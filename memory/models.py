"""Agent Memory 领域模型。

类与数据库表一一对应（表名 = 类名，单数）：
    Session / Conversation / Memory / Summary / Query

约定：
    - 所有 ID 为 UUID 字符串，由默认 DAO 在对象初始化时生成并落库。
    - 对象在 __init__ 时自动落库并取得自己的 ID，外部不传 id / dao。

对外公开的 API：
    session = Session()                    # 自动创建 session + conversation + memory
    session.conversation.add(text, role)   # 记录一条发言
    session.conversation.history()         # 当前未压缩窗口文本
    session.conversation.summarize()       # 未满阈值返回 None，满了返回 Summary
    session.conversation.clear()           # 清空窗口
    session.memory.get_memories()          # 按时间排序的 Summary 列表
    session.memory.add_summary(summary)    # 落库摘要
    session.memory.size()                  # token 数
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

ROLE_MODEL = "model"
ROLE_USER = "user"

# 进程内单调递增序号，用于打破同一微秒内多条记录的先后顺序
_seq = itertools.count()


def now_iso() -> str:
    """统一时间戳格式（微秒 + 单调序号，保证字符串排序 == 实际先后顺序）。

    微秒精度仍可能在极紧的循环里碰撞，因此追加固定宽度序号，
    确保同进程内每条记录严格有序（排序键 `timestamp` 可直接用字符串比较）。
    """
    ts = datetime.now().isoformat(timespec="microseconds")
    return f"{ts}.{next(_seq):06d}"


_qwen_tokenizer = None


def _get_qwen_tokenizer():
    """懒加载 Qwen-14B tokenizer（全局单例）。"""
    global _qwen_tokenizer
    if _qwen_tokenizer is None:
        from importlib import import_module
        from transformers import AutoTokenizer

        try:
            conf = import_module("src.configs.real_config")
        except ImportError:
            conf = import_module("src.configs.config")

        _qwen_tokenizer = AutoTokenizer.from_pretrained(
            conf.Qwen_14B_local_path, trust_remote_code=True
        )
    return _qwen_tokenizer


def _default_tokenizer(text: str) -> int:
    """默认 tokenizer：Qwen-14B tokenizer；加载失败时回退为字符数。"""
    try:
        tok = _get_qwen_tokenizer()
        return len(tok.encode(text))
    except Exception:
        return len(text)


def _dao():
    """返回默认 DAO 单例（延迟导入，避免与 dao 模块循环依赖）。"""
    from dao import get_dao
    return get_dao()


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
@dataclass
class Query:
    """一条原始对话记录（user 或 model 的一次发言）。"""

    conversation_id: str
    role: str                 # 'model' | 'user'
    text: str
    timestamp: str
    token_count: Optional[int] = None
    query_id: Optional[str] = None        # 由 DAO 生成
    summary_id: Optional[str] = None      # 被哪个 summary 汇总（NULL = 未压缩）


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
@dataclass
class Summary:
    """摘要：汇总一批 Query，支持 parent 自引用做二级摘要。"""

    queries: List[Query]
    conversation_id: str
    session_id: str
    memory_id: str
    level: int = 1
    summary_id: Optional[str] = None       # 由 DAO 生成
    content: str = ""                       # generate_content() 生成
    parent_summary_id: Optional[str] = None  # 二级摘要指向父级
    created_at: str = ""

    def generate_content(self, llm: Optional[Callable[[List[Query]], str]] = None) -> str:
        """生成摘要文本。

        llm：可选的可调用对象，接收 self.queries 返回摘要文本；
        未接入时退化为简单拼接（占位实现，正式接入 Qwen 摘要后替换）。
        """
        if llm is not None:
            self.content = llm(self.queries)
        else:
            self.content = " | ".join(f"[{q.role}] {q.text}" for q in self.queries)
        return self.content


# --------------------------------------------------------------------------- #
# Conversation
# --------------------------------------------------------------------------- #
class Conversation:
    """一个对话，对应一个 session；维护未压缩窗口的 token 计数。

    初始化时自动落库并取得 conversation_id（外部不传 id / dao）。
    persist=False 时仅内存运行（用于 _long_conversation 这种无限增长、不落库的对话）。
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        tokenizer: Optional[Callable[[str], int]] = None,
        summary_threshold: int = 2000,
        persist: bool = True,
    ):
        self.tokenizer = tokenizer or _default_tokenizer
        self.summary_threshold = summary_threshold
        self.persist = persist
        # count 是「每个 Conversation 实例内」的状态：当前未压缩窗口的 token 累计
        self.count = 0
        self.pending_queries: List[Query] = []

        if persist:
            dao = _dao()
            self.session_id = session_id or dao.insert_session()
            self.conversation_id = dao.insert_conversation(self.session_id)
        else:
            self.session_id = session_id or ""
            self.conversation_id = "__long__"

    def _count_tokens(self, text: str) -> int:
        try:
            return int(self.tokenizer(text))
        except Exception:
            return _default_tokenizer(text)

    def add(self, text: str, role: str = ROLE_USER) -> Query:
        """记录一条发言：落库（persist 时）+ 累计 token。不自动触发压缩。"""
        token_count = self._count_tokens(text)
        q = Query(
            conversation_id=self.conversation_id,
            role=role,
            text=text,
            timestamp=now_iso(),
            token_count=token_count,
        )
        if self.persist:
            _dao().insert_query(q)
        self.pending_queries.append(q)
        self.count += token_count
        return q

    def history(self) -> str:
        """当前未压缩窗口的 query 拼接文本（按时间排序，用于注入 prompt）。"""
        return "\n".join(f"[{q.role}] {q.text}" for q in self.pending_queries)

    def summarize(self, summarizer: Optional[Callable[[str], str]] = None) -> Optional[Summary]:
        """显式触发摘要：未满阈值返回 None；满了生成 Summary（含 content），清空窗口并返回。

        summarizer：可选的摘要生成回调，接收 history 文本返回摘要文本；
        未传入时退化为拼接（占位）。
        注意：这里只生成 Summary 对象（content 已填充），**不落库**；
        落库交给 Memory.add_summary(summary) 完成。
        """
        if self.count < self.summary_threshold:
            return None
        if not self.pending_queries:
            return None

        history = self.history()
        summary = Summary(
            queries=list(self.pending_queries),
            conversation_id=self.conversation_id,
            session_id=self.session_id,
            memory_id="",  # 由 Memory.add_summary 落库时回填
            level=1,
        )
        if summarizer is not None:
            summary.content = summarizer(history)
        else:
            summary.generate_content()

        # 清空当前窗口
        self.pending_queries = []
        self.count = 0
        return summary

    def clear(self) -> None:
        """清空当前窗口（终局测试：凭记忆重问时使用，数据库历史保留）。"""
        self.pending_queries = []
        self.count = 0


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
class Memory:
    """一个记忆，包含一系列 Summary。初始化时先落库创建记录，再取得自己的 memory_id。"""

    def __init__(
        self,
        session_id: Optional[str] = None,
        tokenizer: Optional[Callable[[str], int]] = None,
    ):
        dao = _dao()
        self.session_id = session_id or dao.insert_session()
        self.memory_id = dao.insert_memory(self.session_id)
        self.tokenizer = tokenizer or _default_tokenizer

    def _count_tokens(self, text: str) -> int:
        try:
            return int(self.tokenizer(text))
        except Exception:
            return _default_tokenizer(text)

    def get_memories(self) -> List[Summary]:
        """返回按时间从早到晚排序的摘要列表。"""
        summaries = _dao().get_summaries_by_memory(self.memory_id)
        summaries.sort(key=lambda s: (s.created_at, s.summary_id or ""))
        return summaries

    def add_summary(self, summary: Summary) -> None:
        """落库一个摘要（含 content），并回填其关联 queries 的 summary_id。"""
        summary.memory_id = self.memory_id
        _dao().insert_summary(summary)

    def size(self) -> int:
        """memory 当前大小：所有摘要 content 的 token 总数。"""
        return sum(self._count_tokens(s.content) for s in self.get_memories())

    def generate(self) -> str:
        """便捷方法：拼接所有摘要文本。"""
        summaries = self.get_memories()
        return "\n".join(s.content for s in summaries)


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
class Session:
    """会话容器：持有 1 个 Conversation 与 1 个 Memory。初始化时自动全部落库。"""

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], int]] = None,
        summary_threshold: int = 2000,
    ):
        self.session_id = _dao().insert_session()
        self.conversation = Conversation(
            self.session_id, tokenizer=tokenizer, summary_threshold=summary_threshold
        )
        self.memory = Memory(self.session_id, tokenizer=tokenizer)
