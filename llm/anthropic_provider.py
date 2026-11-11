"""
Anthropic 在线 API 实现。

默认模型 claude-opus-5-5，走官方 anthropic SDK。
采用流式调用 + get_final_message()，避免长输出触发超时。
"""

from anthropic import Anthropic

from .base import LLMError, LLMProvider


class AnthropicProvider(LLMProvider):
    """联网付费 API 提供方。"""

    def __init__(self, api_key: str, model: str = "claude-opus-5-5"):
        if not api_key:
            raise LLMError(
                "未配置 Anthropic API Key。请二选一：\n"
                "  1) 设置环境变量 ANTHROPIC_API_KEY；\n"
                "  2) 复制 .streamlit/secrets.toml.example 为 .streamlit/secrets.toml 并填入 Key。"
            )
        self.api_key = api_key
        self.model = model
        self._client = Anthropic(api_key=api_key)

    def generate(self, system: str, prompt: str, max_tokens: int) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        with self._client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        # 只取文本块（可能还有 thinking 等其他块，这里一律忽略）
        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        return text
