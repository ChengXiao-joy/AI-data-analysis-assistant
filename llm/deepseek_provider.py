"""DeepSeek 在线 API 实现（OpenAI 兼容接口，直接用 requests 调用）。

说明：不用 openai SDK，而是直接用 requests 调 DeepSeek 的 /chat/completions 接口，
这样可以完全掌控 UTF-8 编码和错误处理，避免 SDK 层的编码问题。
"""

import requests

from .base import LLMError, LLMProvider


class DeepSeekProvider(LLMProvider):
    """DeepSeek 提供方。"""

    def __init__(self, api_key: str, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com"):
        if not api_key:
            raise LLMError(
                "未配置 DEEPSEEK_API_KEY。请在项目根目录 .env 文件中添加：\n"
                "DEEPSEEK_API_KEY=你的key"
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, system: str, prompt: str, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,  # 代码生成用低温度，输出更稳定
        }

        # 发请求（json= 会自动把中文安全地转成 UTF-8）
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.RequestException as e:
            raise LLMError(f"DeepSeek 接口连接失败：{e}")

        # 非 200：尽量解析出友好错误信息
        if resp.status_code != 200:
            err_text = ""
            try:
                err_text = resp.json().get("error", {}).get("message", "")
            except Exception:
                err_text = resp.text
            if resp.status_code in (401, 403):
                raise LLMError("DeepSeek API Key 无效或没有权限，请检查 .env 中的 DEEPSEEK_API_KEY。")
            raise LLMError(f"DeepSeek 接口返回错误（HTTP {resp.status_code}）：{err_text}")

        # 解析返回内容
        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError(f"DeepSeek 返回内容解析失败：{e}")

        return content or ""
