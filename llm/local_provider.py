"""
本地离线模型入口 —— 预留。

V1 只保留这个占位接口，不实现具体本地推理。
后续接入 Ollama / 本地模型时，只需在本文件补全 generate()，上层业务无需改动。
"""

from .base import LLMError, LLMProvider


class LocalProvider(LLMProvider):
    """本地离线模型提供方（预留，V1 未实现具体推理）。"""

    def __init__(self, model: str = ""):
        self.model = model or "local-model"

    def generate(self, system: str, prompt: str, max_tokens: int) -> str:
        raise LLMError(
            "本地离线模型为预留入口，V1 尚未实现。\n"
            "如需使用，请把 config.py 中 CONFIG.llm_provider 改回 'api'。"
        )
