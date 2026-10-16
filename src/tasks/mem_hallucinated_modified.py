"""记忆增强版 Hallucinated Modified 任务。

继承 HalluModified，复用 retrieve_docs / scoring / compute_overall / set_model /
_read_prompt_template；仅把「构造 query」与「发模型」拆成两个方法，方便外部：

1. 拿到 user 的 query 字符串塞进 memory（build_query）；
2. 在「清空 conversation 凭记忆重问」的环节直接生成（generate_from_query，不走 retriever）。
"""
from src.tasks.hallucinated_modified import HalluModified


class MemHalluModified(HalluModified):
    """与 HalluModified 行为一致，但把 prompt 构造与模型调用拆开。"""

    def build_query(self, obj: dict) -> str:
        """构造 prompt（从原 model_generation 拆出的 template.format 部分）。

        返回的是完整的 user 侧 query 字符串，可直接塞进 memory。
        """
        if obj["hallucinatedMod"] == '","msg":"request openai failed"':
            return '","msg":"request openai failed"'
        template = self._read_prompt_template('hallu_mod.txt')
        query = template.format(
            begin=f'{obj["newsBeginning"]}',
            hallu_continue=f'{obj["hallucinatedContinuation"]}',
            search_documents=f'{obj["retrieve_context"]}',
        )
        return query

    def build_question(self, obj: dict) -> str:
        """构造「只有问题、不含检索文档」的 prompt（终局凭记忆重问用）。

        与 build_query 的区别：search_documents 位置为空，模型只能依据记忆回答。
        """
        if obj["hallucinatedMod"] == '","msg":"request openai failed"':
            return '","msg":"request openai failed"'
        template = self._read_prompt_template('hallu_mod.txt')
        query = template.format(
            begin=f'{obj["newsBeginning"]}',
            hallu_continue=f'{obj["hallucinatedContinuation"]}',
            search_documents='（无检索文档，请仅依据你已有的记忆纠正幻觉）',
        )
        return query

    def generate_from_query(self, query: str) -> str:
        """给定已拼好的 prompt，直接生成并解析 real_res。

        用于「清空 conversation 后凭记忆重问最初问题」的环节（不走 retriever）。
        """
        if query == '","msg":"request openai failed"':
            return query
        res = self.model.safe_request(query)
        real_res = res.split('<response>')[-1].split('</response>')[0]
        return real_res.strip()

    def model_generation(self, obj: dict) -> str:
        """只返回 real_res，不做额外处理（保持与 BaseEvaluator 兼容）。"""
        return self.generate_from_query(self.build_query(obj))
