#!/usr/bin/env python3
"""keep_alive 最小验证脚本。

复用 MemoryEvaluator.keep_alive 的核心逻辑：
    - Session() 自建 session + conversation + memory（验证落库 / 关联 / 摘要）
    - _long_conversation 无限增长、永不摘要（用于生成下一个 query）
    - conversation 每满阈值就摘要并清空，摘要入 memory
    - 直到 memory.size() >= memory_threshold 结束

结束时把 _long_conversation.history() 完整倒出到日志文件（只记这一个东西）。
"""
import os
import sys
from datetime import datetime

# 让脚本能 import 项目根目录的 src 包 与 memory 下的模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Session, Conversation, ROLE_USER, ROLE_MODEL
from deepseek_client import DeepSeekClient


def main():
    query_size = 500          # 每轮回复的目标字数
    memory_threshold = 2000   # token 阈值（与主逻辑一致：2k）

    deepseek = DeepSeekClient()

    # 1) 自建 session（验证 session/conversation/memory 三件套）
    session = Session()
    conversation = session.conversation
    memory = session.memory
    print(f"session_id      = {session.session_id}")
    print(f"conversation_id = {conversation.conversation_id}")
    print(f"memory_id       = {memory.memory_id}")

    # 2) 初始 prompt（与 keep_alive 入口一致）
    simulate_query = deepseek.quest("请你结合时下热点，向我抛出一个问题。")
    initial_prompt = (
        f"用不少于{query_size}字回答我下面的问题：{simulate_query}，"
        f"且以后对我的回复的字数也要在{query_size}左右"
    )

    # 3) keep_alive 核心循环（复用原逻辑，threshold=1000）
    _long_conversation = Conversation(persist=False)
    query = initial_prompt
    round_no = 0
    while memory.size() < memory_threshold:
        round_no += 1
        _long_conversation.add(query, role=ROLE_USER)
        conversation.add(query, role=ROLE_USER)

        response = deepseek.request(query)
        if not response or not response.strip():
            raise Exception("DeepSeek returned empty string in keep_alive")
        _long_conversation.add(response, role=ROLE_MODEL)
        conversation.add(response, role=ROLE_MODEL)

        summary = conversation.summarize(summarizer=deepseek.summarize)
        if summary:
            memory.add_summary(summary)
            if memory.size() >= memory_threshold:
                break

        query = deepseek.request(
            f"基于下面的对话，生成一个新的query，不少于{query_size}字\n"
            + _long_conversation.history()
        )
        print(f"[round {round_no}] memory.size()={memory.size()}  summaries={len(memory.get_memories())}")

    # 4) 结束时，把 _long_conversation 的完整 history 倒出到日志文件（只记这一个）
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'session_logs')
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_path = os.path.join(log_dir, f'keep_alive_verify_{ts}.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"# keep_alive 验证（memory_threshold={memory_threshold}, query_size={query_size}）\n")
        f.write(f"# session_id={session.session_id}\n")
        f.write(f"# conversation_id={conversation.conversation_id}\n")
        f.write(f"# memory_id={memory.memory_id}\n")
        f.write(f"# 总轮数={round_no}\n")
        f.write(f"# 最终 memory.size()={memory.size()}\n")
        f.write(f"# 摘要数={len(memory.get_memories())}\n")
        f.write("=" * 80 + "\n")
        f.write(_long_conversation.history())

    print(f"\nDone. long_conversation history dumped to:\n  {log_path}")


if __name__ == "__main__":
    main()
