import datetime
import json
import os
import sys

# 让 memory/ 下的脚本能 import 项目根目录的 evaluator 与 src 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from loguru import logger
from tqdm import tqdm

from evaluator import BaseEvaluator
from models import Session, Conversation, ROLE_USER, ROLE_MODEL
from deepseek_client import DeepSeekClient

FAILED_MARKER = '","msg":"request openai failed"'


class MemoryEvaluator(BaseEvaluator):
    def __init__(
        self,
        task,
        model,
        retriever,
        dataset,
        output_dir: str = './output',
        num_threads: int = 40,
        memory_threshold: int = 2000,
        query_size: int = 500,
        system_prompt: str = (
            "以下是你的记忆区和问题区。请仅依据<记忆>区的内容回答<问题>区的问题，"
            "不要编造记忆中没有的信息。"
        ),
    ):
        super().__init__(task, model, retriever, dataset, output_dir, num_threads)

        self.deepseek = DeepSeekClient()
        self.memory_threshold = memory_threshold
        self.query_size = query_size
        self.system_prompt = system_prompt
        # 注意：session/conversation/memory 不再在 __init__ 全局创建，
        # 而是在 run_memory_loop 里为每个 data_point 独立创建。
        # 收集本次运行产生的所有 session_id（每次 run 独立一份）
        self.session_ids = []
        self.run_id = ""

        # 记忆评估结果输出目录（quest_mem_eval）
        self.mem_output_dir = os.path.join(output_dir, 'quest_mem_eval')
        if not (os.path.exists(self.mem_output_dir) and os.path.isdir(self.mem_output_dir)):
            os.makedirs(self.mem_output_dir)
        self.mem_output_path = os.path.join(
            self.mem_output_dir, f'{self.task.__class__.__name__}_{model.params["model_name"]}.json'
        )

    # ------------------------------------------------------------------ #
    # keep_alive：自对话循环，直到 memory 爆满
    # ------------------------------------------------------------------ #
    def keep_alive(self, query: str, query_size: int):
        """自对话循环，直到 memory 爆满。

        _long_conversation：无限增长、永不摘要的对话（仅内存，用于生成下一个 query）。
        conversation：每次 summarize 后清空窗口的「工作对话」，避免冗余。
        """
        # 无限增长的对话：不落库、不摘要，用于给 deepseek 生成下一个 query
        _long_conversation = Conversation(persist=False)

        while self.memory.size() < self.memory_threshold:
            _long_conversation.add(query, role=ROLE_USER)
            self.conversation.add(query, role=ROLE_USER)

            response = self.deepseek.request(query)
            if not response or not response.strip():
                raise Exception("DeepSeek returned empty string in keep_alive")
            _long_conversation.add(response, role=ROLE_MODEL)
            self.conversation.add(response, role=ROLE_MODEL)

            # conversation 满了就摘要（用 DeepSeek 生成 400 字摘要）
            summary = self.conversation.summarize(summarizer=self._deepseek_summarize)
            if summary:
                # 先写入，再判断是否爆（爆了保留这次写入，不取消）
                self.memory.add_summary(summary)
                if self.memory.size() >= self.memory_threshold:
                    break

            # 基于长对话至今的内容，生成下一个 query
            query = self.deepseek.request(
                f"基于下面的对话，生成一个新的query，不少于{query_size}字\n"
                + _long_conversation.history()
            )

    def _deepseek_summarize(self, history: str) -> str:
        """用 DeepSeek 生成「可重建性」摘要（保留全部关键事实，含不相关话题）。"""
        return self.deepseek.summarize(history)

    # ------------------------------------------------------------------ #
    # run_memory_loop：每个 data_point 视为一轮独立对话
    # ------------------------------------------------------------------ #
    def run_memory_loop(self, data_point, initial_text):
        """① 新建独立 session/conversation/memory → ② 用 initial 回答做 seed
        → ③ 前置初始问题 → ④ keep_alive 直到 memory 爆 → ⑤ 清空 conversation 凭记忆重问。"""
        # 每个问题 / 任务独立一份 session / conversation / memory
        session = Session()
        self.session_ids.append(session.session_id)
        self.conversation = session.conversation
        self.memory = session.memory

        # 断点续跑时 initial 阶段可能跳过该条，retrieve_context 未写入，现场补
        if "retrieve_context" not in data_point or not data_point["retrieve_context"]:
            data_point["retrieve_context"] = self.task.retrieve_docs(data_point)

        # ② seed：用 initial 阶段「基于 retriever 的回答」作为第一次对话，让知识进入记忆。
        query = self.task.build_query(data_point)
        self.conversation.add(query, role=ROLE_USER)
        self.conversation.add(initial_text, role=ROLE_MODEL)

        # ③ 进入 keep_alive 前：让 deepseek 抛出初始问题
        simulate_query = self.deepseek.quest("请你结合时下热点，向我抛出一个问题。")
        initial_prompt = (
            f"用不少于{self.query_size}字回答我下面的问题：{simulate_query}，"
            f"且以后对我的回复的字数也要在{self.query_size}左右"
        )


        self.keep_alive(initial_prompt, self.query_size)

        self.conversation.clear()
        final_prompt = (
            self.system_prompt
            + "\n<记忆>\n"
            + self.memory.generate()
            + "\n</记忆>\n\n<问题>\n"
            + self._initial_question(data_point)
            + "\n</问题>"
        )
        self.conversation.add(final_prompt, role=ROLE_USER)
        final_res = self.task.generate_from_query(final_prompt)
        self.conversation.add(final_res, role=ROLE_MODEL)
        return final_res

    def _initial_question(self, data_point) -> str:
        """终局重问的「问题本身」：不含检索文档，模型只能依据记忆回答。"""
        return self.task.build_question(data_point)

    # ------------------------------------------------------------------ #
    # multithread_batch_scoring：先算 initial 分，再跑 memory loop
    # ------------------------------------------------------------------ #
    def multithread_batch_scoring(self, dataset, sort=True, show_progress_bar=False, contain_original_data=False):
        # 1) initial 分：复用基类多线程逻辑（基线评分，不碰 memory）
        initial_results = super().multithread_batch_scoring(
            dataset, sort, show_progress_bar, contain_original_data
        )

        # 从 initial_results 建 id -> generated_text 映射（含断点续读的磁盘结果）
        initial_text_map = {
            r['id']: r['log']['generated_text'] for r in initial_results
        }

        # 2) memory loop：每个 data_point 独立 session，跑 keep_alive + 终局召回，再评分
        mem_results = []
        for data_point in (tqdm(dataset, desc='memory_loop') if show_progress_bar else dataset):
            initial_text = initial_text_map.get(data_point['ID'])
            if initial_text is None:
                # initial 阶段没产生有效结果（被过滤/失败），跳过 memory loop
                continue
            try:
                final_res = self.run_memory_loop(data_point, initial_text)
                # 过滤无效输出（与基类 process_data_point 一致）
                if final_res == FAILED_MARKER or final_res == '':
                    continue
                data_point["generated_text"] = final_res
                score_dict = self.task.scoring(data_point)
                if score_dict is None:
                    continue
                result = {'id': data_point['ID'], **score_dict}
                if contain_original_data:
                    result['original_data'] = data_point
                mem_results.append(result)
            except Exception as e:
                logger.warning(repr(e))

        self._initial_results = initial_results
        self._mem_results = mem_results
        return sorted(mem_results, key=lambda x: x['id']) if sort else mem_results

    # ------------------------------------------------------------------ #
    # run：复制 BaseEvaluator.run 的保存逻辑，输出到 quest_mem_eval
    # ------------------------------------------------------------------ #
    def run(self, sort=True, show_progress_bar=False, contain_original_data=True):
        # 每次 run 生成独立 run_id（时间戳），用于区分 quick_start 每次跑程序
        self.run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.session_ids = []

        info = {
            'task': self.task.__class__.__name__,
            'llm': str(self.model.params),
        }

        mem_results = self.multithread_batch_scoring(
            self.dataset, sort, show_progress_bar, contain_original_data
        )
        valid_mem = self.remove_invalid(mem_results)
        mem_overall = self.task.compute_overall(valid_mem) if len(valid_mem) > 0 else {}

        initial_valid = self.remove_invalid(self._initial_results)
        initial_overall = self.task.compute_overall(initial_valid) if len(initial_valid) > 0 else {}

        if self.task.use_quest_eval:
            self.lock.acquire()
            self.task.quest_eval.save_quest_gt(self.task.__class__.__name__)
            self.lock.release()

        output = {
            'info': info,
            'overall': mem_overall,
            'results': mem_results,
            'initial_overall': initial_overall,
            'initial_results': self._initial_results,
        }

        with open(self.mem_output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        print(f'Output saved at {self.mem_output_path}!')

        # 保存本次运行产生的所有 session_id 到 txt（每次 run 一个文件，带时间戳）
        self._save_session_ids()
        return output

    def _save_session_ids(self):
        """把本次 run 产生的所有 session_id 写入 txt，文件名带 run_id 区分每次运行。"""
        session_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'session_logs')
        os.makedirs(session_log_dir, exist_ok=True)
        session_log_path = os.path.join(session_log_dir, f'sessions_{self.run_id}.txt')
        with open(session_log_path, 'w', encoding='utf-8') as f:
            for sid in self.session_ids:
                f.write(sid + '\n')
        print(f'Session ids saved at {session_log_path}!')
        return session_log_path
