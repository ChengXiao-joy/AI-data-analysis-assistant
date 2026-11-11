"""
模型调用封装层：对外统一入口。

业务模块统一使用：
    from llm import llm_generate, LLMError
"""

from .base import (
    LLMError,
    LLMProvider,
    create_provider,
    get_provider,
    reset_provider,
    llm_generate,
)

__all__ = [
    "llm_generate",
    "LLMError",
    "LLMProvider",
    "create_provider",
    "get_provider",
    "reset_provider",
]
