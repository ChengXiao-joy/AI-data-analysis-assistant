"""
配置层 —— 集中管理应用的所有可调参数。

设计思路：
- 所有"魔法值"（模型名、上传上限、质检阈值、图表引擎等）集中在这一处。
- API Key 不在代码里硬编码：
    1) 先从项目根目录 .env 文件加载到环境变量（DEEPSEEK_API_KEY）；
    2) 再依次从 环境变量 → Streamlit secrets → secrets.toml 读取。
- 切换大模型：改 llm_provider（"deepseek" / "api"(Anthropic) / "local"）。
"""

import os
from dataclasses import dataclass


# 启动时加载项目根目录的 .env（把 DEEPSEEK_API_KEY 等写进环境变量）
try:
    from dotenv import load_dotenv
    _BASE = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_BASE, ".env"))
except Exception:
    pass


@dataclass
class AppConfig:
    # ---------- 大模型 ----------
    llm_provider: str = "deepseek"          # "deepseek"=DeepSeek(默认)；"api"=Anthropic；"local"=本地预留
    api_model: str = "claude-opus-5-5"      # Anthropic 模型 ID
    deepseek_model: str = "deepseek-chat"   # DeepSeek 模型 ID
    deepseek_base_url: str = "https://api.deepseek.com"  # DeepSeek API 地址
    local_model: str = ""                   # 本地模型名（预留）
    max_tokens: int = 8192                  # 单次生成最大输出 token 数
    api_key: str = ""                       # 手动指定（一般留空，走自动解析）

    # ---------- 数据 ----------
    max_upload_mb: int = 100             # 上传文件大小上限（MB）
    max_sample_rows: int = 5             # 传给大模型的样本行数上限（≤5）
    preview_rows: int = 10               # 页面预览显示的行数

    # ---------- 质检 ----------
    missing_rate_alert: float = 0.30     # 字段缺失率超过该值 → 标红提示
    duplicate_rate_alert: float = 0.30   # 重复行占比超过该值 → 标红提示

    # ---------- 图表 ----------
    chart_engine: str = "plotly"         # "plotly"=默认交互式；"matplotlib"=静态兜底

    # ---------- 本地暂存 ----------
    temp_dir: str = ".tmp"

    # ---------- 密钥解析 ----------
    def anthropic_key(self) -> str:
        if self.api_key:
            return self.api_key.strip()
        return _get_secret("ANTHROPIC_API_KEY")

    def deepseek_key(self) -> str:
        return _get_secret("DEEPSEEK_API_KEY")

    def api_key_resolved(self) -> str:
        """兼容旧调用：返回 Anthropic Key。"""
        return self.anthropic_key()


def _get_secret(name: str) -> str:
    """按优先级取密钥：环境变量 → Streamlit secrets → secrets.toml。"""
    val = os.environ.get(name, "").strip()
    if val:
        return val
    val = _read_streamlit_secret(name)
    if val:
        return val
    return _read_secrets_toml(name)


def _read_streamlit_secret(name: str) -> str:
    try:
        import streamlit as st
    except Exception:
        return ""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return ""


def _read_secrets_toml(name: str) -> str:
    try:
        import tomllib
    except ImportError:
        return ""
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, ".streamlit", "secrets.toml")
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return str(data.get(name, "") or "").strip()
    except Exception:
        return ""


# 全局配置单例：业务模块统一 import 它使用
CONFIG = AppConfig()
