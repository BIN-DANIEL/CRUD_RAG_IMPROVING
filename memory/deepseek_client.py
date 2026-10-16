"""DeepSeek 客户端（OpenAI 兼容接口）。

用于记忆实验的「自对话」环节：
    quest(prompt)  ：让模型「抛出问题」（模拟 user 的提问）。
    request(query) ：普通对话补全（模型回答）。

API Key 从 src/configs 读取（DeepSeek_api_key），base_url 指向官方兼容端点。
"""
from importlib import import_module

import openai
from loguru import logger

try:
    conf = import_module("src.configs.real_config")
except ImportError:
    conf = import_module("src.configs.config")


class DeepSeekClient:
    def __init__(
        self,
        model_name: str = "deepseek-flash",
        summarize_model_name: str = "deepseek-v4-pro",
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        top_p: float = 0.9,
    ):
        self.model_name = model_name
        self.summarize_model_name = summarize_model_name
        self.api_key = api_key or getattr(conf, "DeepSeek_api_key", "")
        self.base_url = base_url
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        # reasoning 模型的思考 token 也计入 max_tokens，摘要需给足余量
        self.summarize_max_tokens = 8192

    def _call(self, model: str, query: str, max_tokens: int = None) -> str:
        openai.api_key = self.api_key
        openai.base_url = self.base_url
        res = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_new_tokens,
            top_p=self.top_p,
        )
        return res.choices[0].message.content

    def request(self, query: str) -> str:
        return self._call(self.model_name, query)

    def quest(self, prompt: str) -> str:
        """让模型抛出一个问题（用于模拟 user 的提问）。"""
        return self.request(prompt)

    def summarize(self, history: str) -> str:
        """将对话历史压缩为「可重建性」摘要。

        标准：一个只看到这份摘要的人，能够准确重建过去发生了什么，不遗漏关键信息。
        对话中出现多个互不相关的话题时，必须全部保留，不得因主题无关而省略。
        """
        prompt = (
            "你需要把一段对话历史压缩成摘要。\n\n"
            "【黄金标准】当旁人阅读到这份摘要时，能够重建过去发生了什么，"
            "而不会有信息的遗漏——包括所有重要的信息。\n\n"
            "请遵循以下标准：\n"
            "1. 保留所有关键事实：人名、时间、地点、数字、具体事件、任务要求。\n"
            "2. 对话中如果出现多个互不相关的话题，必须全部保留，"
            "不得因为某个话题与主线无关而省略。\n"
            "3. 不要只提炼「主题观点」，要保留「具体发生了什么事」。\n\n"
            "【示例】\n"
            "历史对话：\n"
            "[user] 帮我把明天下午3点的项目评审会加进日程，地点在A栋201会议室。\n"
            "[model] 好的，已记录：明天下午3点，A栋201会议室，项目评审会。\n"
            "[user] 对了，我家猫\"煤球\"最近得了肠胃炎，医生说每天要喂两次药。\n"
            "[model] 记下了，煤球需要每天喂两次肠胃炎的药。\n\n"
            "摘要：\n"
            "本次对话包含两件互不相关的事：1）用户安排了一场明天下午3点、"
            "地点在A栋201会议室的项目评审会；2）用户的猫\"煤球\"患上肠胃炎，需每天喂两次药。"
            "两件事无关联，但都是需记住的关键信息。\n\n"
            "现在请对以下对话生成摘要（约400字，同样遵循「可重建性」标准，"
            "保留全部关键事实）：\n"
            "----------------------以下为你需要摘要的内容-------------------------------\n\n"
            f"{history}"
        )
        # 摘要用 V4 Pro 模型（默认已开启 reasoning，思考 token 计入 max_tokens，需加大余量）
        return self._call(self.summarize_model_name, prompt, max_tokens=self.summarize_max_tokens)

    def safe_request(self, query: str) -> str:
        try:
            return self.request(query)
        except Exception as e:
            logger.warning(repr(e))
            return ""
