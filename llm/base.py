"""
模型调用封装层 —— 核心。

这一层把"调用哪个大模型、怎么调用"全部收口，业务模块只调用统一的
llm_generate() 函数，不需要关心底层是 Anthropic 在线 API 还是本地模型。

关键设计：
1. LLMProvider：抽象接口，所有厂商实现同一个 generate()。
2. create_provider()：工厂，按 CONFIG.llm_provider 决定用哪个实现。
3. llm_generate()：全局统一入口，含异常包装，业务代码唯一使用的函数。
4. 切换模型：改 config.py 里 llm_provider，或调用 reset_provider()。
"""

from abc import ABC, abstractmethod
from typing import Optional

import config


class LLMError(Exception):
    """大模型调用相关的错误统一包装成它，上层只需 catch 这一种。"""


class LLMProvider(ABC):
    """大模型提供方抽象接口：所有实现必须提供 generate()。"""

    @abstractmethod
    def generate(self, system: str, prompt: str, max_tokens: int) -> str:
        """生成文本。

        system     : 系统提示词（角色设定、规则约束）
        prompt     : 用户输入内容
        max_tokens : 最大输出 token 数
        返回       : 生成的文本
        """


def create_provider(settings: "config.AppConfig" = None) -> LLMProvider:
    """按配置创建大模型提供方（工厂）。"""
    settings = settings or config.CONFIG

    if settings.llm_provider == "deepseek":
        from .deepseek_provider import DeepSeekProvider
        return DeepSeekProvider(
            api_key=settings.deepseek_key(),
            model=settings.deepseek_model,
            base_url=settings.deepseek_base_url,
        )

    if settings.llm_provider == "api":
        try:
            from .anthropic_provider import AnthropicProvider
        except ImportError as e:
            raise LLMError("未安装 anthropic 依赖，请先执行：pip install -r requirements.txt") from e
        return AnthropicProvider(api_key=settings.api_key_resolved(), model=settings.api_model)

    if settings.llm_provider == "local":
        from .local_provider import LocalProvider
        return LocalProvider(model=settings.local_model)

    raise LLMError(f"未知的 llm_provider：{settings.llm_provider!r}（可选 'deepseek' / 'api' / 'local'）")


# 默认提供方缓存：延迟创建，配置变化后调 reset_provider() 重建
_default_provider: Optional[LLMProvider] = None


def get_provider() -> LLMProvider:
    """获取当前默认提供方（首次调用时创建并缓存）。"""
    global _default_provider
    if _default_provider is None:
        _default_provider = create_provider()
    return _default_provider


def reset_provider() -> None:
    """配置变更后调用，丢弃缓存的提供方，下次调用按新配置重建。"""
    global _default_provider
    _default_provider = None


def llm_generate(
    prompt: str,
    system: str = "",
    max_tokens: Optional[int] = None,
    provider: Optional[LLMProvider] = None,
) -> str:
    """统一的大模型调用入口 —— 业务模块唯一应使用的函数。

    会做三件事：
    1. 选提供方（默认当前配置的提供方）；
    2. 兜底 max_tokens；
    3. 捕获底层异常，统一转成 LLMError，方便界面友好提示。
    """
    provider = provider or get_provider()
    if max_tokens is None:
        max_tokens = config.CONFIG.max_tokens

    try:
        return provider.generate(system=system, prompt=prompt, max_tokens=max_tokens)
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"大模型调用失败：{exc}") from exc
