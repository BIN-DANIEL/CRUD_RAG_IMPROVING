#!/usr/bin/env python3
"""端到端演示：Session 自动落库 → add → summarize → add_summary → get_memories / size。"""
from models import Session, ROLE_USER, ROLE_MODEL


def main():
    # Session() 自动创建 session + conversation + memory（均落库，ID 自生成）
    session = Session(summary_threshold=20)
    conv = session.conversation
    mem = session.memory

    print(f"session_id      = {session.session_id}")
    print(f"conversation_id = {conv.conversation_id}")
    print(f"memory_id       = {mem.memory_id}")
    print("-" * 50)

    # 模拟对话：每轮 add 之后显式 summarize
    rounds = [
        (ROLE_USER, "帮我查一下上个月的销售数据"),
        (ROLE_MODEL, "好的，上个月销售额为 120 万"),
        (ROLE_USER, "那环比增长了多少？"),
        (ROLE_MODEL, "环比增长 8.5%"),
    ]
    for role, text in rounds:
        q = conv.add(text, role=role)
        print(f"add  [{q.role:5s}] {text!r}  token={q.token_count}  count={conv.count}")

        summary = conv.summarize()       # 未满阈值返回 None
        if summary:
            mem.add_summary(summary)
            print(f"  -> 触发摘要压缩，生成 summary，关联 {len(summary.queries)} 条 query")

    print("-" * 50)

    # history()：当前未压缩窗口文本
    print("conversation.history():")
    print(conv.history())
    print("-" * 50)

    # get_memories() / size()
    summaries = mem.get_memories()
    print(f"mem.get_memories() 摘要数量 = {len(summaries)}")
    for s in summaries:
        print(f"  level={s.level}  content = {s.content}")
    print(f"mem.size() = {mem.size()}（token 数，默认 tokenizer 按字符估算）")
    print("-" * 50)

    # Memory.generate()：拼接摘要文本（用于注入 prompt）
    print("Memory.generate() 输出:")
    print(mem.generate())


if __name__ == "__main__":
    main()
