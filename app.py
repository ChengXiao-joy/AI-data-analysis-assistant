"""AI 数据分析助手 V1 —— 应用主入口。

页面结构：单页面 + 三个标签页（st.tabs），对应三大核心模块：
  1. 数据上传与预览
  2. 数据质检报告
  3. 智能问答分析

本步骤（步骤2）只搭骨架：保证页面切换正常、数据能跨页面共享。
每个页面真正的业务逻辑，在步骤3 里逐个填充。
"""

import streamlit as st

import state
from modules import upload, quality, qa


# 必须是第一个 Streamlit 命令
st.set_page_config(page_title="AI 数据分析助手 V1", page_icon="📊", layout="wide")

# 应用启动时初始化全局状态（上传数据 / 质检结果 / 对话历史）
state.init_state()

st.title("AI 数据分析助手 V1")
st.caption("本地运行 · 上传即质检 · 自然语言问答，无需写代码")

# 让三个 tab 吸顶：页面上下滚动时，标签栏始终停留在顶部（Streamlit 自带 header 下方）
st.markdown(
    """
    <style>
        div[role="tablist"] {
            position: sticky;
            top: 2.875rem;
            z-index: 999;
            background-color: #ffffff;
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e0e0e0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 三个标签页
tab_upload, tab_quality, tab_qa = st.tabs(
    ["📁 数据上传与预览", "🔍 数据质检报告", "💬 智能问答分析"]
)

with tab_upload:
    upload.render()

with tab_quality:
    quality.render()

with tab_qa:
    qa.render()
