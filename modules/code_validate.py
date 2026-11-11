"""代码前置校验层 —— 大模型生成代码后、执行前，先做校验。

校验顺序（任一不通过 → 不执行，直接返回分类化中文提示）：
  ① 列名校验：代码引用的列名是否真实存在于表；
  ② 样本量校验：分组后是否存在样本量 < 5 的组；
  ③ 语法与逻辑校验：代码语法是否正确、是否给 result 赋值。

说明：本层是启发式校验（正则 + AST），目标是拦住"明显错误"，不追求 100% 精确。
"""

import ast
import re

import pandas as pd


# 校验失败的分类
CATEGORY_COLUMN = "字段不存在"
CATEGORY_SAMPLE = "样本过少"
CATEGORY_LOGIC = "逻辑风险"


def validate(code: str, df: pd.DataFrame):
    """对生成的代码做前置校验。

    返回 (分类, 中文提示)；通过时返回 (None, None)。
    """
    # ① 列名
    err = _check_columns(code, df)
    if err:
        return CATEGORY_COLUMN, err

    # ② 样本量
    err = _check_sample_size(code, df)
    if err:
        return CATEGORY_SAMPLE, err

    # ③ 语法与逻辑
    err = _check_syntax(code)
    if err:
        return CATEGORY_LOGIC, err
    err = _check_logic(code)
    if err:
        return CATEGORY_LOGIC, err

    return None, None


def _extract_column_refs(code: str) -> list:
    """提取代码里可能作为列名引用的字符串（启发式，覆盖常见 pandas 写法）。"""
    refs = []
    # df["列"] / df['列']
    refs += re.findall(r"""df\s*\[\s*['"]([^'"]+)['"]""", code)
    # groupby("列") / sort_values("列") / 等常见方法
    refs += re.findall(
        r"""\.(?:groupby|sort_values|nlargest|nsmallest|value_counts)\s*\([^'"]*['"]([^'"]+)['"]""",
        code,
    )
    # [ '列' ] / [ "列" ]（如 groupby(...)['列'] 的取值）
    refs += re.findall(r"""\[\s*['"]([^'"]+)['"]""", code)

    # 去重保序
    seen = set()
    out = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _check_columns(code: str, df: pd.DataFrame):
    """① 校验代码引用的列名是否都存在于表。"""
    refs = _extract_column_refs(code)
    missing = [r for r in refs if r not in df.columns]
    if missing:
        return f"代码引用了不存在的字段：{', '.join(missing)}。请检查字段名是否写错。"
    return None


def _check_sample_size(code: str, df: pd.DataFrame):
    """② 校验是否存在分组后样本量 < 5 的组。"""
    group_cols = re.findall(r"""\.groupby\s*\([^'"]*['"]([^'"]+)['"]""", code)
    problems = []
    for col in group_cols:
        if col not in df.columns:
            continue  # 列不存在已在①处理
        counts = df[col].value_counts()
        small = counts[counts < 5]
        if len(small) > 0:
            top = small.nsmallest(3)
            desc = "、".join(f"{k}（{v} 条）" for k, v in top.items())
            problems.append(f"字段「{col}」分组后有 {len(small)} 个组样本量 < 5（如 {desc}）")
    if problems:
        return "分组后样本量过少，统计结果可能不稳定：" + "；".join(problems)
    return None


def _check_syntax(code: str):
    """③a 校验代码语法是否正确。"""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"代码存在语法错误：{e}"
    return None


def _check_logic(code: str):
    """③b 校验明显逻辑问题：是否给 result 赋值（否则拿不到结果）。"""
    if not re.search(r"\bresult\s*=", code):
        return "代码没有给 result 赋值，无法返回结果。"
    return None
