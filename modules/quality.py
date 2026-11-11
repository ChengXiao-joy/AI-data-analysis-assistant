"""页2：数据质检报告。

从全局状态读取数据，懒计算（并缓存）质检结果后展示。
所有计算都是本地 pandas，不调用大模型、不需要 API Key。
"""

import pandas as pd
import streamlit as st

import state
from modules import data_quality


def _fmt_ratio(r) -> str:
    """把 0~1 的比例格式化成百分比字符串。"""
    return f"{float(r):.1%}"


def render() -> None:
    st.subheader("数据质检报告")

    if not state.has_dataset():
        st.warning("请先在「数据上传与预览」页载入数据。")
        return

    df = state.get_dataset()

    # 懒计算 + 缓存：结果没算过就算一次，之后复用，避免每次刷新重算
    report = state.get_quality_report()
    if report is None:
        report = data_quality.run_quality_check(df)
        state.set_quality_report(report)

    # —— 异常提示（红色高亮） ——
    for alert in report["alerts"]:
        st.error(f"⚠️ {alert['message']}")
    if not report["alerts"]:
        st.success("未发现异常：缺失率、重复率、异常值、类型冲突都在可接受范围。")

    # —— 基础信息 ——
    st.markdown("### 📋 基础信息")
    basic = report["basic"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("行数", f"{basic['rows']:,}")
    c2.metric("列数", str(basic["cols"]))
    c3.metric("缺失单元格", str(int(sum(report["missing"]["counts"].values()))))
    c4.metric("重复行数", str(report["duplicates"]["count"]))

    st.markdown("**字段类型：**")
    st.dataframe(
        data_quality.index_from_one(pd.DataFrame({
            "字段": list(basic["field_types"].keys()),
            "类型": list(basic["field_types"].values()),
        })),
        width="stretch",
    )

    # —— 缺失值统计 ——
    st.markdown("### 🕳️ 缺失值统计")
    missing = report["missing"]
    st.dataframe(
        data_quality.index_from_one(pd.DataFrame({
            "字段": list(missing["counts"].keys()),
            "缺失数量": [int(missing["counts"][c]) for c in missing["counts"]],
            "缺失占比": [_fmt_ratio(missing["ratios"][c]) for c in missing["counts"]],
        })),
        width="stretch",
    )

    # —— 重复行 ——
    st.markdown("### 🔁 重复行统计")
    d = report["duplicates"]
    st.write(f"重复行数：**{d['count']}** 行，占比 **{_fmt_ratio(d['ratio'])}**")

    # —— 数值字段统计 ——
    if report["numeric"]:
        st.markdown("### 🔢 数值字段统计")
        st.dataframe(
            data_quality.index_from_one(pd.DataFrame([
                {
                    "字段": col,
                    "最大值": round(s["max"], 2),
                    "最小值": round(s["min"], 2),
                    "均值": round(s["mean"], 2),
                    "中位数": round(s["median"], 2),
                    "标准差": round(s["std"], 2),
                }
                for col, s in report["numeric"].items()
            ])),
            width="stretch",
        )
    else:
        st.info("无数值字段。")

    # —— 文本字段唯一值 ——
    if report["text_unique"]:
        st.markdown("### 🏷️ 文本字段唯一值数量")
        st.dataframe(
            data_quality.index_from_one(pd.DataFrame({
                "字段": list(report["text_unique"].keys()),
                "唯一值数量": list(report["text_unique"].values()),
            })),
            width="stretch",
        )
    else:
        st.info("无文本字段。")

    # —— 数值字段 3σ 异常值 ——
    if report["outliers"]:
        st.markdown("### ⚠️ 数值字段 3σ 异常值")
        st.dataframe(
            data_quality.index_from_one(pd.DataFrame([
                {
                    "字段": col,
                    "异常值个数": info["count"],
                    "占比": _fmt_ratio(info["ratio"]),
                    "正常下限": round(info["low"], 2),
                    "正常上限": round(info["high"], 2),
                }
                for col, info in report["outliers"].items()
            ])),
            width="stretch",
        )
    else:
        st.info("无数值字段可做 3σ 异常值检测。")

    # —— 字段类型冲突 ——
    if report["type_conflicts"]:
        st.markdown("### 🔀 字段类型冲突")
        for col, info in report["type_conflicts"].items():
            st.warning(
                f"字段「{col}」混入数值与文本：{info['numeric_like']}/{info['total']} "
                f"个为数值（{info['ratio']:.0%}）。"
            )
    else:
        st.info("未发现字段类型冲突。")

    # —— 清洗建议 ——
    st.markdown("### 🧹 清洗建议（仅建议，不修改原数据）")
    if report["suggestions"]:
        for s in report["suggestions"]:
            st.markdown(f"- **{s['target']}**：{s['issue']} → {s['suggestion']}")
    else:
        st.success("数据较干净，暂无需清洗。")
