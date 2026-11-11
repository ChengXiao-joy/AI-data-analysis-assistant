"""本地数据质检 —— 纯 pandas 计算，不涉及大模型。

对外提供一个函数 run_quality_check(df)，返回结构化质检结果 dict。
展示层在 quality.py，计算层在这里，方便单独测试。
"""

import pandas as pd

import config


def _classify_dtype(dtype) -> str:
    """把 pandas 数据类型归类成人类可读的中文标签。"""
    if pd.api.types.is_bool_dtype(dtype):
        return "布尔"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "日期"
    if pd.api.types.is_integer_dtype(dtype):
        return "整数"
    if pd.api.types.is_float_dtype(dtype):
        return "小数"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return "文本"
    return str(dtype)


def _is_numeric(dtype) -> bool:
    """是否为可做统计的数值列（排除布尔）。"""
    if pd.api.types.is_bool_dtype(dtype):
        return False
    return pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype)


def _is_text(dtype) -> bool:
    """是否为文本列。"""
    return pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype)


def _classify_field_role(col_name, series) -> str:
    """识别字段的业务角色：标识字段 / 维度字段 / 指标字段 / 文本字段。

    规则（启发式，按顺序判断）：
    - 标识字段：列名含 id / 编号 / 编码 / 单号（如订单ID、用户ID）
    - 指标字段：数值类型（可度量，如销售额、数量）
    - 维度字段：低基数短文本 或 日期（用于分组，如地区、产品、时间）
    - 文本字段：长文本 或 列名含备注/评论/说明等（自由文本）
    """
    name = str(col_name)
    low = name.lower().replace(" ", "").replace("_", "")
    dtype = series.dtype

    # 标识字段
    if ("id" in low) or ("编号" in name) or ("编码" in name) or ("单号" in name):
        return "标识字段"

    # 指标字段（数值，排除布尔）
    if (pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype)) \
            and not pd.api.types.is_bool_dtype(dtype):
        return "指标字段"

    # 日期 → 维度字段（时间维度）
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "维度字段"

    # 文本类：先看列名是否明显是"自由文本"，再看取值平均长度
    if any(k in name for k in ("备注", "评论", "说明", "描述", "留言", "意见", "地址", "note", "comment", "remark")):
        return "文本字段"

    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        s = series.dropna().astype(str)
        if len(s) and s.str.len().mean() > 15:
            return "文本字段"
        return "维度字段"

    return "文本字段"  # 兜底


def index_from_one(obj):
    """返回行号从 1 开始的副本（仅展示用，不改原数据）。"""
    out = obj.copy()
    out.index = range(1, len(out) + 1)
    return out


def _detect_outliers(series: pd.Series, sigma: float = 3.0):
    """数值列的 3σ 异常值检测：返回 {count, ratio, low, high}；无法计算返回 None。"""
    s = series.dropna()
    if len(s) == 0:
        return None
    mean = s.mean()
    std = s.std()
    if pd.isna(std) or std == 0:
        return None  # 值全相同，无法用 3σ 判断
    low = mean - sigma * std
    high = mean + sigma * std
    count = int(((s < low) | (s > high)).sum())
    return {
        "count": count,
        "ratio": count / len(s),
        "low": float(low),
        "high": float(high),
    }


def _detect_type_conflict(series: pd.Series):
    """检测文本列里是否混入"像数字"的值（即类型冲突）。

    返回 {numeric_like, total, ratio} 或 None（无冲突）。
    """
    if not _is_text(series.dtype):
        return None
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    converted = pd.to_numeric(non_null, errors="coerce")
    numeric_like = int(converted.notna().sum())
    ratio = numeric_like / len(non_null)
    if 0 < ratio < 1:  # 一部分像数字、一部分不像 → 类型冲突
        return {"numeric_like": numeric_like, "total": int(len(non_null)), "ratio": ratio}
    return None


def _build_suggestions(missing, duplicates, outliers, type_conflicts) -> list:
    """根据质检结果生成清洗建议（只建议，不修改原数据）。"""
    suggestions = []
    for col, ratio in missing["ratios"].items():
        cnt = int(missing["counts"][col])
        if cnt <= 0:
            continue
        if ratio >= config.CONFIG.missing_rate_alert:
            suggestions.append({
                "target": f"字段「{col}」",
                "issue": f"缺失率 {ratio:.1%}（偏高）",
                "suggestion": "建议删除该列，或用众数/中位数填充。",
            })
        else:
            suggestions.append({
                "target": f"字段「{col}」",
                "issue": f"缺失 {cnt} 个（{ratio:.1%}）",
                "suggestion": "可用中位数/众数填充，或按分析需要保留缺失。",
            })

    if duplicates["count"] > 0:
        suggestions.append({
            "target": "整张表",
            "issue": f"重复行 {duplicates['count']} 行（{duplicates['ratio']:.1%}）",
            "suggestion": "建议去重（保留首条），避免重复计数。",
        })

    for col, info in outliers.items():
        if info["count"] > 0:
            suggestions.append({
                "target": f"字段「{col}」",
                "issue": f"3σ 异常值 {info['count']} 个（{info['ratio']:.1%}）",
                "suggestion": "建议核查异常值来源，可剔除或单独分析。",
            })

    for col, info in type_conflicts.items():
        suggestions.append({
            "target": f"字段「{col}」",
            "issue": f"混入数值与文本（{info['ratio']:.0%} 为数值）",
            "suggestion": "建议统一该列类型，或把异常值单独处理。",
        })
    return suggestions


def run_quality_check(df: pd.DataFrame) -> dict:
    """对 df 做本地质检，返回结构化结果。

    结果结构：
        basic          {rows, cols, field_types}
        missing        {counts, ratios}
        duplicates     {count, ratio}
        numeric        {字段: {max, min, mean, median, std}}
        text_unique    {字段: 唯一值数量}
        outliers       {字段: {count, ratio, low, high}}      # 3σ 异常值
        type_conflicts {字段: {numeric_like, total, ratio}}    # 类型冲突
        suggestions    [ {target, issue, suggestion}, ... ]    # 清洗建议
        alerts         [ {type, col?, ratio, message}, ... ]   # 高亮风险项
    """
    rows, cols = df.shape

    # 1) 字段类型
    field_types = {col: _classify_dtype(df[col].dtype) for col in df.columns}

    # 2) 缺失值
    missing_counts = df.isna().sum().to_dict()
    missing_ratios = {
        col: (int(n) / rows if rows else 0.0)
        for col, n in missing_counts.items()
    }

    # 3) 重复行
    dup_count = int(df.duplicated().sum())
    dup_ratio = dup_count / rows if rows else 0.0

    # 4) 数值字段统计
    numeric = {}
    for col in df.columns:
        if not _is_numeric(df[col].dtype):
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        numeric[col] = {
            "max": float(s.max()),
            "min": float(s.min()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
        }

    # 5) 文本字段唯一值
    text_unique = {}
    for col in df.columns:
        if not _is_text(df[col].dtype):
            continue
        text_unique[col] = int(df[col].dropna().nunique())

    # 6) 数值字段 3σ 异常值
    outliers = {}
    for col in df.columns:
        if not _is_numeric(df[col].dtype):
            continue
        info = _detect_outliers(df[col])
        if info is not None:
            outliers[col] = info

    # 7) 字段类型冲突
    type_conflicts = {}
    for col in df.columns:
        info = _detect_type_conflict(df[col])
        if info is not None:
            type_conflicts[col] = info

    # 8) 清洗建议
    suggestions = _build_suggestions(
        {"counts": missing_counts, "ratios": missing_ratios},
        {"count": dup_count, "ratio": dup_ratio},
        outliers,
        type_conflicts,
    )

    # 9) 异常标记（高亮风险项）
    alerts = []
    for col, ratio in missing_ratios.items():
        if ratio >= config.CONFIG.missing_rate_alert:
            alerts.append({
                "type": "missing", "col": col, "ratio": ratio,
                "message": f"字段「{col}」缺失率 {ratio:.1%}，超过阈值 {config.CONFIG.missing_rate_alert:.0%}。",
            })
    if dup_ratio >= config.CONFIG.duplicate_rate_alert:
        alerts.append({
            "type": "duplicate", "ratio": dup_ratio,
            "message": f"重复行占比 {dup_ratio:.1%}，超过阈值 {config.CONFIG.duplicate_rate_alert:.0%}。",
        })
    for col, info in outliers.items():
        if info["count"] > 0:
            alerts.append({
                "type": "outlier", "col": col, "ratio": info["ratio"],
                "message": f"字段「{col}」3σ 异常值 {info['count']} 个（{info['ratio']:.1%}）。",
            })
    for col, info in type_conflicts.items():
        alerts.append({
            "type": "type_conflict", "col": col, "ratio": info["ratio"],
            "message": f"字段「{col}」存在类型冲突（{info['ratio']:.0%} 为数值、其余为文本）。",
        })

    return {
        "basic": {"rows": rows, "cols": cols, "field_types": field_types},
        "missing": {"counts": missing_counts, "ratios": missing_ratios},
        "duplicates": {"count": dup_count, "ratio": dup_ratio},
        "numeric": numeric,
        "text_unique": text_unique,
        "outliers": outliers,
        "type_conflicts": type_conflicts,
        "suggestions": suggestions,
        "alerts": alerts,
    }
