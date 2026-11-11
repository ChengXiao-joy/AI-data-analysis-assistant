"""全局状态管理 —— 统一管理跨页面共享的数据。

设计思路：
- 所有跨页面共享的数据都通过这里读写，键名集中定义，避免各处硬编码。
- init_state() 在应用启动时调用一次，初始化所有状态。
- 三个页面（上传/质检/问答）各自通过 get/set 函数读写，不直接碰 session_state。
- 状态清单：
    dataset           上传的数据集（pandas.DataFrame 或 None）
    file_meta         文件信息（文件名、行列数、列名、内存等 dict）
    quality_report    质检结果（dict 或 None，懒计算后缓存）
    chat_history      对话历史（list[dict]，每项 {"role":..., "content":...}）
    last_processed_file  最近一次处理的上传文件标识（用于避免重复读取）
"""

import streamlit as st


KEY_DATASET = "dataset"
KEY_FILE_META = "file_meta"
KEY_QUALITY = "quality_report"
KEY_CHAT = "chat_history"
KEY_LAST_FILE = "last_processed_file"


def init_state() -> None:
    """应用启动时调用一次，初始化所有全局状态（已存在则跳过）。"""
    defaults = {
        KEY_DATASET: None,
        KEY_FILE_META: None,
        KEY_QUALITY: None,
        KEY_CHAT: [],
        KEY_LAST_FILE: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_new_dataset(df, meta: dict) -> None:
    """加载一份新数据集：写入数据+元信息，并清空旧质检结果和对话。"""
    st.session_state[KEY_DATASET] = df
    st.session_state[KEY_FILE_META] = meta
    st.session_state[KEY_QUALITY] = None
    st.session_state[KEY_CHAT] = []


def clear_all() -> None:
    """清空所有数据，回到初始状态。"""
    st.session_state[KEY_DATASET] = None
    st.session_state[KEY_FILE_META] = None
    st.session_state[KEY_QUALITY] = None
    st.session_state[KEY_CHAT] = []
    st.session_state[KEY_LAST_FILE] = None


# ---------- 数据集 ----------
def get_dataset():
    return st.session_state.get(KEY_DATASET)


def has_dataset() -> bool:
    return st.session_state.get(KEY_DATASET) is not None


# ---------- 文件信息 ----------
def get_file_meta():
    return st.session_state.get(KEY_FILE_META)


# ---------- 质检结果 ----------
def get_quality_report():
    return st.session_state.get(KEY_QUALITY)


def set_quality_report(report) -> None:
    st.session_state[KEY_QUALITY] = report


# ---------- 对话历史 ----------
def get_chat_history():
    return st.session_state.get(KEY_CHAT)


def add_chat_message(role: str, content: str) -> None:
    """追加一条纯文本对话（role: user / assistant）。"""
    st.session_state[KEY_CHAT].append({"role": role, "content": content})


def add_chat_entry(entry: dict) -> None:
    """追加一条结构化对话记录（问答页使用，可含代码/结果/图表/错误）。"""
    st.session_state[KEY_CHAT].append(entry)


def clear_chat() -> None:
    st.session_state[KEY_CHAT] = []


# ---------- 上传文件跟踪 ----------
def get_last_file_id():
    return st.session_state.get(KEY_LAST_FILE)


def set_last_file_id(file_id) -> None:
    st.session_state[KEY_LAST_FILE] = file_id
