"""页1：数据上传与预览。

支持上传 csv / xlsx，全程本地读取、本地计算，不调用大模型。
上传后把 DataFrame 存入全局状态，三个 tab 共享同一份数据。
"""

import html
import io

import charset_normalizer
import numpy as np
import pandas as pd
import streamlit as st

import config
import state
from errors import EmptyDataError, EncodingError, FileFormatError, friendly_message
from modules.data_quality import _classify_field_role, index_from_one


# 常见的 csv 编码候选（依次尝试；不含 latin-1，否则无法识别真正的编码错误）
_ENCODING_CANDIDATES = ["utf-8-sig", "gb18030", "gbk", "utf-8"]

# 「全部合并」选项的显示文字（多 sheet 时出现在下拉框里）
_ALL_SHEETS_LABEL = "（全部）合并所有工作表"


def _detect_encoding(raw: bytes) -> str:
    """用 charset-normalizer 自动识别文件编码，识别失败回退 utf-8。"""
    try:
        best = charset_normalizer.from_bytes(raw).best()
        if best and best.encoding:
            return best.encoding
    except Exception:
        pass
    return "utf-8"


def _read_csv(raw: bytes) -> pd.DataFrame:
    """读取 csv：先按检测到的编码读，失败再依次尝试常见中文编码。

    编码都不对时抛 EncodingError；编码对但内容无法解析时抛 FileFormatError。
    """
    if not raw.strip():
        raise EmptyDataError("文件内容为空，请上传包含数据的文件。")

    candidates = []
    detected = _detect_encoding(raw)
    if detected:
        candidates.append(detected)
    candidates += _ENCODING_CANDIDATES

    seen = set()
    for enc in candidates:
        if enc in seen:
            continue
        seen.add(enc)
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue  # 编码不对，试下一个
        except Exception as e:
            raise FileFormatError(f"CSV 内容无法解析（可能不是有效的 CSV 文件）：{e}") from e
    raise EncodingError("无法识别该 CSV 文件的编码，请另存为 UTF-8 编码后重新上传。")


def _read_excel(raw: bytes, sheet_name=0) -> pd.DataFrame:
    """读取 xlsx 的指定工作表（默认第一张；内部已是 UTF-8，无需编码处理）。"""
    try:
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl", sheet_name=sheet_name)
    except Exception as e:
        raise FileFormatError(f"Excel 文件无法解析（可能已损坏或不是有效的 xlsx）：{e}") from e


def _read_sheet_names(raw: bytes) -> list:
    """读取 xlsx 的所有工作表名称（只读结构，不加载全部数据）。"""
    try:
        excel = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
        return list(excel.sheet_names)
    except Exception as e:
        raise FileFormatError(f"Excel 文件无法解析（可能已损坏或不是有效的 xlsx）：{e}") from e


def _read_all_sheets(raw: bytes) -> pd.DataFrame:
    """读取 xlsx 的所有工作表，纵向合并成一张表，并加一列「来源工作表」标记来源。"""
    try:
        excel = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
        frames = []
        for name in excel.sheet_names:
            part = excel.parse(sheet_name=name)
            part.insert(0, "来源工作表", name)
            frames.append(part)
        return pd.concat(frames, ignore_index=True)
    except Exception as e:
        raise FileFormatError(f"合并工作表时出错：{e}") from e


def _demo_df() -> pd.DataFrame:
    """模拟销售数据集：含少量缺失值、2 条重复行，方便演示质检效果。"""
    rng = np.random.default_rng(42)  # 固定随机种子，结果可复现
    n = 30
    df = pd.DataFrame({
        "订单ID": [f"ORD{1000 + i}" for i in range(n)],
        "用户ID": [f"U{1000 + i // 2}" for i in range(n)],
        "地区": rng.choice(["华东", "华南", "华北", "西南"], n),
        "产品": rng.choice(["手机", "电脑", "平板", "耳机"], n),
        "数量": rng.integers(1, 20, n),
        "销售额": rng.integers(500, 5000, n).astype(float),
        "备注": ["正常"] * n,
        "下单日期": pd.date_range("2026-01-01", periods=n),
    })
    # 缺失值：销售额 3 处、地区 2 处、备注 12 处（备注缺失率高，用于演示标红）
    df.loc[[3, 7, 12], "销售额"] = np.nan
    df.loc[[5, 18], "地区"] = None
    df.loc[:11, "备注"] = None
    # 重复行：追加 2 条重复记录
    df = pd.concat([df, df.iloc[[2, 4]]], ignore_index=True)
    return df


def _fmt_bytes(n: int) -> str:
    """把字节数转成易读的 KB / MB。"""
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _find_user_id_column(df: pd.DataFrame):
    """找出「用户ID」列（启发式：列名含 用户/客户 且含 id/编号/编码，或直接叫 id）。"""
    for col in df.columns:
        if not isinstance(col, str):
            continue
        low = col.lower().replace(" ", "").replace("_", "")
        if low in ("id", "userid", "customerid", "用户id", "客户id"):
            return col
        if ("用户" in col or "客户" in col) and ("id" in low or "编号" in col or "编码" in col):
            return col
    return None


def _find_time_column(df: pd.DataFrame):
    """找出时间/月份列（优先 datetime 类型，其次列名含 日期/时间/月份/date/month）。"""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
            return col
    for col in df.columns:
        if not isinstance(col, str):
            continue
        low = col.lower().replace(" ", "").replace("_", "")
        if any(k in low for k in ("日期", "时间", "月份", "年月", "date", "time", "month")):
            return col
    return None


def _time_range_str(df: pd.DataFrame) -> str:
    """时间范围字符串：时间列的最小值～最大值；无时间列返回 '—'。"""
    col = _find_time_column(df)
    if col is None:
        return "—"
    s = df[col].dropna()
    if len(s) == 0:
        return "—"
    lo, hi = s.min(), s.max()
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        # 同一个月就展示到日，跨月展示到月
        if lo.year == hi.year and lo.month == hi.month:
            return f"{lo:%Y-%m-%d} ~ {hi:%Y-%m-%d}"
        return f"{lo:%Y-%m} ~ {hi:%Y-%m}"
    return f"{lo} ~ {hi}"


def _render_missing_warning(df: pd.DataFrame) -> None:
    """字段列表上方的简易预警栏：展示缺失率 TOP3（>20% 红、10~20% 黄）。"""
    rows = len(df)
    ratios = {}
    for col in df.columns:
        r = df[col].isna().sum() / rows if rows else 0.0
        if r > 0:
            ratios[col] = r
    top3 = sorted(ratios.items(), key=lambda kv: -kv[1])[:3]
    if not top3:
        return

    parts = []
    for col, r in top3:
        if r > 0.20:
            color = "#d93025"  # 红
        elif r >= 0.10:
            color = "#f0a500"  # 黄
        else:
            color = "#5f6368"  # 灰
        parts.append(
            f'<span style="color:{color};font-weight:600;">{html.escape(str(col))}（{r:.1%}）</span>'
        )

    st.markdown(
        '<div style="background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;padding:8px 12px;margin:6px 0;">'
        '<b>⚠️ 缺失率 TOP3：</b>' + "　".join(parts)
        + '<span style="color:#888;font-size:0.85rem;">　完整质检报告请切换至【数据质检报告】标签</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def _store_df(df: pd.DataFrame, filename: str, sheet_names=None, sheet=None) -> None:
    """把 DataFrame 和元信息写入全局状态（同时清掉旧质检结果和对话）。"""
    if df is None or df.empty:
        raise EmptyDataError("数据为空（0 行），请确认文件里有实际数据。")
    meta = {
        "filename": filename,
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "memory": int(df.memory_usage(deep=True).sum()),
        "sheet_names": sheet_names,  # 多 sheet 的 xlsx 才有值；csv/单表为 None
        "sheet": sheet,              # 当前加载的工作表名
    }
    state.load_new_dataset(df, meta)


def _dismiss_uploader(uploaded) -> None:
    """载入示例/清空后，让上传组件下一轮复位为空，避免旧文件残留被重新读取。"""
    if uploaded is not None:
        state.set_last_file_id(f"{uploaded.name}|{uploaded.size}")
    st.session_state["_reset_uploader"] = True


def render() -> None:
    st.subheader("数据上传与预览")

    # 上一轮点了「载入示例/清空」，这里在创建上传组件前先把它复位为空
    if st.session_state.get("_reset_uploader", False):
        st.session_state["file_uploader"] = None
        st.session_state["_reset_uploader"] = False

    version = st.session_state.get("_uploader_version", 0)
    uploaded = st.file_uploader(
        f"上传 csv 或 xlsx 文件（单个 ≤ {config.CONFIG.max_upload_mb}MB）",
        type=["csv", "xlsx"],
        key=f"file_uploader_{version}",
    )

    # 只处理"新出现"的文件，避免每次刷新重复读取
    if uploaded is not None:
        if uploaded.size > config.CONFIG.max_upload_mb * 1024 * 1024:
            st.error(
                f"文件 {uploaded.size / 1024 / 1024:.1f} MB 超过 "
                f"{config.CONFIG.max_upload_mb} MB 上限，请换一个更小的文件。"
            )
        else:
            file_id = f"{uploaded.name}|{uploaded.size}"
            if state.get_last_file_id() != file_id:
                try:
                    raw = uploaded.getvalue()
                    if uploaded.name.lower().endswith(".csv"):
                        df = _read_csv(raw)
                        _store_df(df, uploaded.name)
                    else:
                        # xlsx：识别所有工作表，默认读第一张；多 sheet 时可切换
                        sheet_names = _read_sheet_names(raw)
                        if not sheet_names:
                            raise FileFormatError("该 Excel 文件没有有效的工作表。")
                        df = _read_excel(raw, sheet_name=sheet_names[0])
                        _store_df(df, uploaded.name, sheet_names=sheet_names, sheet=sheet_names[0])
                        st.session_state["sheet_selector"] = sheet_names[0]  # 重置选择器
                    state.set_last_file_id(file_id)
                    st.success(f"已读取「{uploaded.name}」")
                except Exception as e:
                    st.error(friendly_message(e))

    # —— 多工作表选择器（当前 xlsx 有多个 sheet 时显示）——
    meta = state.get_file_meta() or {}
    sheet_names = meta.get("sheet_names") or []
    if len(sheet_names) > 1:
        options = [_ALL_SHEETS_LABEL] + sheet_names
        current = meta.get("sheet") or sheet_names[0]
        if current not in options:
            current = sheet_names[0]
        selected = st.selectbox("选择工作表", options, key="sheet_selector")
        if selected != current:
            # 用户切换了工作表 / 选了"合并分析"
            try:
                if selected == _ALL_SHEETS_LABEL:
                    df = _read_all_sheets(uploaded.getvalue())
                    _store_df(df, uploaded.name, sheet_names=sheet_names, sheet=_ALL_SHEETS_LABEL)
                    st.success("已合并所有工作表（新增「来源工作表」列）")
                else:
                    df = _read_excel(uploaded.getvalue(), sheet_name=selected)
                    _store_df(df, uploaded.name, sheet_names=sheet_names, sheet=selected)
                    st.success(f"已切换到工作表「{selected}」")
            except Exception as e:
                st.error(friendly_message(e))
            else:
                st.rerun()

    # —— 示例数据 / 清空 ——
    col_btn, col_clear = st.columns([1, 1])
    with col_btn:
        if st.button("载入示例数据"):
            _store_df(_demo_df(), "示例销售数据.csv")
            _dismiss_uploader(uploaded)
            st.success("示例销售数据已载入，可切到「数据质检报告」查看。")

    with col_clear:
        if st.button("清空数据"):
            state.clear_all()
            _dismiss_uploader(uploaded)
            st.warning("已清空数据。")

    # —— 预览区 ——
    st.divider()
    if not state.has_dataset():
        st.info("尚未上传数据。可上传文件，或点「载入示例数据」体验。")
        return

    df = state.get_dataset()
    meta = state.get_file_meta() or {}

    st.markdown("### 📄 数据预览")

    # 文件名（保留，作为一行小字）
    st.caption(f"文件：{meta.get('filename', '-')}")

    # 指标卡片：原 3 个 + 新增 3 个（有效样本数 / 唯一用户数 / 数据时间范围）
    valid_count = int(df.dropna(how="all").shape[0])
    id_col = _find_user_id_column(df)
    unique_users = int(df[id_col].dropna().nunique()) if id_col else None
    time_range = _time_range_str(df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("行数", f"{len(df):,}")
    c2.metric("列数", str(len(df.columns)))
    c3.metric("占用内存", _fmt_bytes(meta.get("memory", 0)))
    c4.metric("有效样本数", f"{valid_count:,}")
    c5.metric("唯一用户数", f"{unique_users:,}" if unique_users is not None else "—")
    c6.metric("数据时间范围", time_range)

    if meta.get("sheet") == _ALL_SHEETS_LABEL:
        st.caption("已合并所有工作表，新增「来源工作表」列标记每行来自哪张表。")

    # 简易预警栏（字段列表上方）
    _render_missing_warning(df)

    # 字段列表（新增「字段类型标签」列）
    st.markdown("**📋 字段列表：**")
    st.dataframe(
        index_from_one(pd.DataFrame({
            "字段": list(df.columns),
            "类型": [str(t) for t in df.dtypes],
            "字段类型标签": [_classify_field_role(c, df[c]) for c in df.columns],
        })),
        width="stretch",
    )

    st.markdown("**🔍 前 10 行数据：**")
    st.dataframe(index_from_one(df.head(config.CONFIG.preview_rows)), width="stretch")
