"""页3：智能问答分析。

流程：用户提问 → 把 schema + 样本发给大模型生成 pandas 代码 → 本地沙盒执行 → 展示结果。
- 只上传 schema（字段名/类型）+ 前 5 行样本，不传全量数据。
- 代码在受限命名空间里执行，禁止读写文件、联网、修改系统。
- 结果固定分区：【数据质检摘要】【文字结论】【图表】【推测与局限说明】。
"""

import html
import io

import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免在 Windows 上弹窗口
# 配置中文字体，避免图表里的中文（如"华东""华南"）显示成方框
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方框的问题
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
import state
from errors import friendly_message, execution_message
from llm import llm_generate, LLMError
from modules import code_validate
from modules.data_quality import _classify_dtype, index_from_one


# —— 系统提示词：让模型只输出可执行的 pandas 代码 ——
_SYSTEM_PROMPT = """你是数据分析助手，负责把用户的自然语言问题翻译成可执行的 pandas 代码。

环境里已准备好这些变量，直接使用，不要重新 import：
- df：要分析的数据表（pandas DataFrame）
- pd：pandas
- np：numpy
- plt：matplotlib.pyplot

规则：
1. 只做查询、统计、绘图，禁止读写文件、联网、修改系统。
2. 把最终结果赋给变量 result：
   - 表格类结果 → pandas DataFrame
   - 单个数值/字符串 → 直接赋值（int/float/str）
3. 需要绘图时用 matplotlib，把图形对象赋给变量 fig，并遵守：
   - 必须设置中文标题（ax.set_title）、坐标轴标签（ax.set_xlabel / ax.set_ylabel）和图例（ax.legend）
   - 按数据特征自动选图型：x 轴是分类字段 → 柱状图(bar)；是时间/日期 → 折线图(line)；是占比/构成 → 饼图(pie)
4. 数值必须来自 pandas 计算结果，禁止自行推算或补造任何数字。
5. 不要写 print、不要写解释文字、不要用 markdown 代码块，只输出纯 Python 代码。
"""

# —— 业务解读 & 推测的系统提示词 ——
_INTERP_SYSTEM_PROMPT = """你是业务分析助手，根据给定的数据统计结果，输出一段「业务解读 & 推测」。

规则：
1. 先简短复述数据事实（只陈述统计结果里能看到的，不夸大、不添油加醋）。
2. 再给出业务层面的可能性推测，用"可能 / 推测 / 或许"等措辞，不要当成确定结论。
3. 如果现有字段不足以支撑推测，直接说明"数据不足，无法进一步推测"，不要编造理由。
4. 只引用统计结果里实际出现的数字，禁止自行推算或补造数字。
5. 保持简洁，3~6 句话，用中文。
6. 不要重复罗列完整明细数字，只提炼要点。
"""

# 对话气泡样式：圆角 + 浅色底，让「你 / AI」两侧更清晰
_CHAT_CSS = """
<style>
    [data-testid="stChatMessageContent"] {
        border-radius: 12px;
        padding: 6px 12px;
        background-color: #f8f9fa;
        border: 1px solid #eceff1;
    }
</style>
"""


def _noop(*args, **kwargs):
    """占位 print，忽略模型的调试输出。"""


# 沙盒里的安全内置函数（不含 open/eval/exec/__import__ 等危险项）
_SAFE_BUILTINS = {
    "len": len, "range": range, "sorted": sorted, "sum": sum,
    "min": min, "max": max, "abs": abs, "round": round,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "enumerate": enumerate, "zip": zip, "isinstance": isinstance,
    "print": _noop,
}


def _build_prompt(df: pd.DataFrame, question: str) -> str:
    """构建发给大模型的提示词：schema + 样本 + 问题（不传全量数据）。"""
    schema_lines = [f"- {col}: {_classify_dtype(df[col].dtype)}" for col in df.columns]
    schema = "\n".join(schema_lines)
    sample = df.head(config.CONFIG.max_sample_rows).to_string(index=False)
    return (
        f"数据表字段 schema（字段名: 类型）：\n{schema}\n\n"
        f"前 {config.CONFIG.max_sample_rows} 行样本：\n{sample}\n\n"
        f"用户问题：{question}"
    )


def _extract_code(raw: str) -> str:
    """去掉模型可能加上的 ```python ... ``` 围栏。"""
    text = (raw or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _validate_code(code: str):
    """安全校验：返回错误信息（None 表示通过）。"""
    for line in code.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            return f"检测到 import 语句，已禁止：{s}"
    for bad in ["open(", "eval(", "exec(", "compile(", "__import__",
                "os.", "sys.", "subprocess", "shutil", "socket",
                "pickle", "builtins", "getattr(", "setattr(",
                "globals(", "locals(", "__"]:
        if bad in code:
            return f"检测到危险代码片段「{bad}」，已禁止。"
    return None


def _run_sandbox(code: str, df: pd.DataFrame):
    """在受限命名空间里执行代码，返回 (result, 图表PNG字节, 错误信息)。"""
    err = _validate_code(code)
    if err:
        return None, None, err

    plt.close("all")  # 清掉上一轮可能残留的图
    namespace = {
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "__builtins__": _SAFE_BUILTINS,
    }

    try:
        exec(code, namespace)
    except Exception as e:
        return None, None, execution_message(e)

    result = namespace.get("result")
    fig = namespace.get("fig")
    if fig is None and plt.get_fignums():
        fig = plt.gcf()

    fig_bytes = None
    if fig is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        fig_bytes = buf.getvalue()
    plt.close("all")  # 转成 PNG 后清理，释放内存
    return result, fig_bytes, None


def _classify_result(result) -> str:
    if isinstance(result, pd.DataFrame):
        return "dataframe"
    if isinstance(result, pd.Series):
        return "series"
    if result is None:
        return "none"
    return "scalar"


def _build_quality_summary(df: pd.DataFrame) -> str:
    """生成简短的数据质检摘要（本地计算，不调用大模型）。"""
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    dup = int(df.duplicated().sum())
    return f"本次分析基于 {rows} 行 × {cols} 列数据；其中缺失值 {missing} 个，重复行 {dup} 行。"


def _format_result_for_interpretation(result, kind: str) -> str:
    """把计算结果格式化成简洁文本，用于喂给大模型做业务解读（只传统计结果，不传原始明细）。"""
    if kind == "dataframe":
        text = result.to_string(index=False)
    elif kind == "series":
        text = result.to_string()
    elif kind == "scalar":
        text = str(result)
    else:
        return "（本次没有返回表格/数值结果，可能只做了绘图）"
    # 限制长度，避免把过大的结果发给大模型
    if len(text) > 2000:
        text = text[:2000] + "\n…（结果过长，已截断）"
    return text


def _generate_interpretation(question: str, result, result_kind: str) -> str:
    """第二次调用大模型：基于「已算好的统计结果」生成业务解读 & 推测。"""
    facts = _format_result_for_interpretation(result, result_kind)
    prompt = f"用户问题：{question}\n\n计算得到的统计结果：\n{facts}"
    return llm_generate(prompt=prompt, system=_INTERP_SYSTEM_PROMPT, max_tokens=1024)


def _render_speculation_section(text, error) -> None:
    """渲染「推测与局限说明」板块（浅色背景，与客观结果明显区分）。"""
    st.divider()
    st.markdown("**【推测与局限说明】**")
    st.caption("【AI 推测，仅供业务参考，不构成决策依据】")
    if error:
        st.warning(f"解读生成失败：{error}")
    elif not text or not text.strip():
        st.info("（本次没有生成推测）")
    else:
        safe_text = html.escape(text).replace("\n", "<br>")
        st.markdown(
            f'<div style="background-color:#f7f7f7;padding:10px 14px;'
            f'border-radius:8px;border-left:4px solid #e0a800;">{safe_text}</div>',
            unsafe_allow_html=True,
        )


def _store_exchange(question: str, entry: dict) -> None:
    """把一次问答（用户提问 + 助手回复）写入对话历史。"""
    state.add_chat_entry({"role": "user", "content": question})
    state.add_chat_entry(entry)


def _handle_question(question: str, speculation_enabled: bool) -> None:
    """处理一次提问：生成代码 → 沙盒执行 →（可选）业务解读 → 存入对话历史。

    用 st.status 展示分阶段进度动画，让用户清楚看到当前处于哪一步。
    """
    df = state.get_dataset()
    quality_summary = _build_quality_summary(df)

    # 默认给一条"失败"回复，成功时再覆盖，保证任何情况都有反馈
    entry = {
        "role": "assistant", "code": "", "result": None,
        "result_kind": "error", "chart": None, "error": "处理失败，请重试。",
        "interpretation": None, "interp_error": None,
        "quality_summary": quality_summary,
    }

    with st.status("已收到你的问题", expanded=True) as status:
        # 阶段 1：大模型生成代码
        status.update(label="🤔 大模型正在思考、生成 pandas 代码…（一般 5~20 秒，请稍候）", state="running")
        try:
            raw_code = llm_generate(
                prompt=_build_prompt(df, question),
                system=_SYSTEM_PROMPT,
                max_tokens=4096,
            )
        except LLMError as e:
            status.update(label="❌ 生成失败", state="error")
            entry["error"] = str(e)
        except Exception as e:
            status.update(label="❌ 出错", state="error")
            entry["error"] = friendly_message(e)
        else:
            code = _extract_code(raw_code)
            if not code:
                status.update(label="❌ 生成失败", state="error")
                entry["error"] = "模型没有返回有效代码，请重试。"
            else:
                # 阶段 1.5：代码前置校验（先不执行，拦住字段不存在/样本过少/逻辑风险）
                status.update(label="🔍 正在校验代码（字段名 / 样本量 / 语法）…", state="running")
                category, val_err = code_validate.validate(code, df)
                if val_err:
                    # 校验不通过：不执行，直接返回分类化中文提示
                    status.update(label="❌ 校验不通过", state="error")
                    entry = {
                        "role": "assistant", "code": code, "result": None,
                        "result_kind": "error", "chart": None,
                        "error": f"【{category}】{val_err}",
                        "interpretation": None, "interp_error": None,
                        "quality_summary": quality_summary,
                    }
                else:
                    # 阶段 2：校验通过才执行
                    status.update(label="✅ 校验通过，正在本地执行…", state="running")
                    result, fig_bytes, exec_err = _run_sandbox(code, df)

                    interpretation = None
                    interp_error = None

                    # 阶段 3（可选）：基于统计结果生成业务解读 & 推测
                    if exec_err is None and speculation_enabled:
                        status.update(label="💡 正在生成业务推测…", state="running")
                        try:
                            interpretation = _generate_interpretation(
                                question, result, _classify_result(result)
                            )
                        except LLMError as e:
                            interp_error = str(e)
                        except Exception as e:
                            interp_error = friendly_message(e)

                    if exec_err:
                        status.update(label="⚠️ 执行出错", state="error")
                    else:
                        status.update(label="✅ 完成", state="complete")

                    entry = {
                        "role": "assistant",
                        "code": code,
                        "result": result,
                        "result_kind": _classify_result(result) if exec_err is None else "error",
                        "chart": fig_bytes,
                        "error": exec_err,
                        "interpretation": interpretation,
                        "interp_error": interp_error,
                        "quality_summary": quality_summary,
                    }

    # 写入对话历史
    _store_exchange(question, entry)


def _render_entry(entry: dict) -> None:
    """渲染一条对话记录。结果区固定四分区：质检摘要 → 文字结论 → 图表 → 推测与局限。"""
    if entry["role"] == "user":
        with st.chat_message("user"):
            st.markdown("**👤 你**")
            st.write(entry["content"])
        return

    with st.chat_message("assistant"):
        st.markdown("**🤖 AI 助手**")
        # 思考过程（面试加分：展示生成的代码）
        if entry.get("code"):
            st.markdown("**生成的代码：**")
            st.code(entry["code"], language="python")

        # 出错时只显示友好错误，不显示分区
        if entry.get("error"):
            st.error(f"⚠️ {entry['error']}")
            return

        # 分区 1：【数据质检摘要】
        if entry.get("quality_summary"):
            st.markdown("**【数据质检摘要】**")
            st.write(entry["quality_summary"])

        # 分区 2：【文字结论】
        result = entry.get("result")
        kind = entry.get("result_kind")
        st.markdown("**【文字结论】**")
        if kind in ("dataframe", "series"):
            st.dataframe(index_from_one(result), width="stretch")
        elif kind == "scalar":
            if isinstance(result, (int, float, bool)):
                st.metric("结果", result)
            else:
                st.write(result)
        else:  # none
            st.info("代码执行成功，但没有返回表格/数值结果（可能只做了绘图）。")

        # 分区 3：【图表】
        if entry.get("chart"):
            st.markdown("**【图表】**")
            st.image(entry["chart"])

        # 分区 4：【推测与局限说明】
        interp = entry.get("interpretation")
        interp_err = entry.get("interp_error")
        if interp is not None or interp_err is not None:
            _render_speculation_section(interp, interp_err)


def _render_example_chips() -> None:
    """展示示例问题按钮，点击后把完整问题填入下方输入框（方便快速体验）。"""
    examples = [
        ("各地区平均销售额", "统计各地区的平均销售额"),
        ("各地区销售额柱状图", "画各地区销售额的柱状图"),
        ("销售额最高3条", "销售额最高的3条记录"),
    ]
    cols = st.columns(len(examples))
    for i, (label, question) in enumerate(examples):
        if cols[i].button(label, key=f"example_chip_{i}"):
            st.session_state["qa_input"] = question


def render() -> None:
    st.subheader("智能问答分析")
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)

    has_data = state.has_dataset()
    has_key = bool(config.CONFIG.deepseek_key())

    # —— 前置检查提示 ——
    if not has_data:
        st.warning("请先上传/载入示例数据")
    elif not has_key:
        st.warning("请配置 API Key 后使用问答功能")

    # —— 顶栏：清空对话 + 业务推测开关 ——
    col_clear, col_toggle = st.columns([1, 2])
    with col_clear:
        if has_data and st.button("清空对话"):
            state.clear_chat()
            st.rerun()
    with col_toggle:
        speculation_enabled = st.toggle(
            "开启业务推测（AI 解读）",
            value=True,
            key="speculation_enabled",
            help="关闭后只返回客观计算结果，不再调用大模型生成业务解读。",
        )

    # —— 对话历史 ——
    if has_data:
        history = state.get_chat_history()
        if not history:
            st.info("输入问题开始分析，或点下方示例快速体验。")
            if has_data and has_key:
                _render_example_chips()
        else:
            for entry in history:
                _render_entry(entry)

    # —— 输入框（无数据或无 Key 时禁用）——
    enabled = has_data and has_key
    with st.form("qa_form", clear_on_submit=True):
        user_input = st.text_input(
            "输入问题",
            placeholder="例如：统计每个地区的平均销售额",
            key="qa_input",
            disabled=not enabled,
        )
        submitted = st.form_submit_button("提交", disabled=not enabled)

    if submitted and enabled:
        if not user_input.strip():
            st.warning("请输入问题。")
        else:
            _handle_question(user_input.strip(), speculation_enabled)
            st.rerun()
