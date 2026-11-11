"""临时验收脚本：验证「配置层 + 模型调用层」是否跑通。

用法（在项目根目录执行）：
    python selfcheck_llm.py
"""

import config
from llm import llm_generate, LLMError


def main() -> None:
    print("=" * 52)
    print("配置自检")
    print("=" * 52)
    print(f"llm_provider    : {config.CONFIG.llm_provider}")
    if config.CONFIG.llm_provider == "deepseek":
        print(f"deepseek_model  : {config.CONFIG.deepseek_model}")
        key = config.CONFIG.deepseek_key()
        key_name = "DEEPSEEK_API_KEY"
    else:
        print(f"api_model       : {config.CONFIG.api_model}")
        key = config.CONFIG.api_key_resolved()
        key_name = "ANTHROPIC_API_KEY"
    print(f"max_tokens      : {config.CONFIG.max_tokens}")
    print(f"chart_engine    : {config.CONFIG.chart_engine}")
    print(f"max_sample_rows : {config.CONFIG.max_sample_rows}")

    if key:
        print(f"API Key         : 已配置（前 6 位 {key[:6]}...）")
    else:
        print(f"API Key         : 未配置（{key_name}）")
    print("=" * 52)

    print("\n调用大模型（生成一句自我介绍）……")
    try:
        reply = llm_generate(
            prompt="请用一句话介绍你自己。",
            system="你是「AI 数据分析助手」的底层大模型。",
        )
        print("\n大模型回复：")
        print(reply)
        print("\n[OK] 模型调用封装层跑通。")
    except LLMError as e:
        print(f"\n[提示] 调用未成功：{e}")
        print("请先在项目根目录 .env 中配置 DEEPSEEK_API_KEY 后重试（见 .env.example）。")


if __name__ == "__main__":
    main()
