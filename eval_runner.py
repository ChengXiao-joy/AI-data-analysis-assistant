"""评测脚本 —— 独立于页面代码，驱动真实分析链路跑 testcases.json。

【这是什么】
本脚本不依赖 Streamlit 页面，直接调用底层纯函数（读取表格 → 大模型生成代码 →
前置校验 → 沙盒执行），把每个 case 的客观结果采集下来，供你人工判定。

【重要原则】
- 脚本只负责「执行 + 采集」，不自动判定对错。
- 结果 CSV 里的「是否通过」列留空，由你根据「预期校验结果/预期要点」自己填。

【怎么用】
1. 准备 testcases.json（格式见 testcases.example.json），字段：
     case_id        用例编号（任意字符串）
     表路径         要分析的表格路径（csv/xlsx，相对项目根目录或绝对路径）
     用户提问       自然语言问题
     预期校验结果   你预期它应通过 / 或应被哪类校验拦截（仅供你判定参考，脚本不比对）
     预期要点       你预期结果应包含的要点（仅供你判定参考，脚本不比对）
2. 确保项目根目录有 .env（配置了 DEEPSEEK_API_KEY），因为要真调大模型。
3. 运行：
     python eval_runner.py testcases.json
   （不带参数时默认读项目根目录下的 testcases.json）

【结果在哪】
- eval_results/results.csv  逐条结果（含留空的「是否通过」列）
- eval_results/summary.txt  汇总统计（代码成功率/校验拦截率/平均耗时/失败清单）
"""

import csv
import json
import os
import sys
import time

# 让脚本能 import 项目里的 config / llm / modules
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from llm import llm_generate, LLMError
from modules import code_validate
from modules.upload import _read_csv, _read_excel
from modules.qa import (
    _SYSTEM_PROMPT,
    _build_prompt,
    _extract_code,
    _run_sandbox,
    _classify_result,
)


# 结果 CSV 的列（「是否通过」留空给你填）
CSV_HEADER = [
    "case_id", "表路径", "用户提问", "预期校验结果", "预期要点",
    "实际校验结果", "是否执行成功", "错误类型", "实际输出摘要", "耗时(秒)", "是否通过",
]


def load_table(path: str):
    """根据文件后缀加载表格（复用上传模块的读取函数）。"""
    if path.endswith(".csv"):
        with open(path, "rb") as f:
            return _read_csv(f.read())
    if path.endswith((".xlsx", ".xls")):
        with open(path, "rb") as f:
            return _read_excel(f.read())
    raise ValueError(f"不支持的文件类型：{path}")


def summarize_output(out) -> str:
    """把执行结果摘要成一段短文本（用于记录，不做判断）。"""
    kind = _classify_result(out)
    if kind == "dataframe":
        return f"表格 {out.shape[0]}行×{out.shape[1]}列：{out.to_string(index=False)[:300]}"
    if kind == "series":
        return f"序列 {len(out)} 项：{out.to_string()[:300]}"
    if kind == "scalar":
        return str(out)[:300]
    return "无返回结果（可能只做了绘图）"


def run_one_case(case: dict) -> dict:
    """跑一个 case，返回采集到的客观结果字典。异常不中断，记录到 error_type。"""
    case_id = case.get("case_id", "?")
    table = case.get("表路径") or case.get("table", "")
    question = case.get("用户提问") or case.get("question", "")

    result = {
        "case_id": case_id,
        "table": table,
        "question": question,
        "expected_validation": case.get("预期校验结果", ""),
        "expected_points": case.get("预期要点", ""),
        "actual_validation": "",   # 实际校验结果：通过 / 字段不存在 / 样本过少 / 逻辑风险
        "exec_success": "否",      # 是否执行成功
        "error_type": "",          # 错误类型（无错误则为空）
        "actual_output": "",       # 实际输出摘要
        "elapsed": 0.0,
        # 内部统计标志（不写进 CSV，只用于汇总）
        "_validation_triggered": False,
        "_exec_success": False,
    }

    t0 = time.time()
    try:
        # 1) 加载表格
        table_path = table if os.path.isabs(table) else os.path.join(PROJECT_ROOT, table)
        df = load_table(table_path)

        # 2) 生成代码
        prompt = _build_prompt(df, question)
        raw_code = llm_generate(prompt=prompt, system=_SYSTEM_PROMPT, max_tokens=4096)
        code = _extract_code(raw_code)
        if not code:
            result["error_type"] = "空代码"
            result["actual_output"] = "模型未返回有效代码"
            result["elapsed"] = time.time() - t0
            return result

        # 3) 前置校验
        category, val_err = code_validate.validate(code, df)
        if val_err:
            result["_validation_triggered"] = True
            result["actual_validation"] = category  # 字段不存在 / 样本过少 / 逻辑风险
            result["error_type"] = "校验拦截"
            result["actual_output"] = val_err
            result["elapsed"] = time.time() - t0
            return result

        # 4) 校验通过才执行
        result["actual_validation"] = "通过"
        out, fig_bytes, exec_err = _run_sandbox(code, df)
        if exec_err:
            result["error_type"] = "执行失败"
            result["actual_output"] = exec_err
            result["elapsed"] = time.time() - t0
            return result

        # 5) 成功
        result["_exec_success"] = True
        result["exec_success"] = "是"
        result["actual_output"] = summarize_output(out)
        result["elapsed"] = time.time() - t0
        return result

    except LLMError as e:
        result["error_type"] = "LLM生成失败"
        result["actual_output"] = str(e)
    except Exception as e:
        result["error_type"] = "表加载/流程失败"
        result["actual_output"] = f"{type(e).__name__}: {e}"

    result["elapsed"] = time.time() - t0
    return result


def write_results(results: list, out_path: str) -> None:
    """把逐条结果写成 CSV（utf-8-sig，Excel 可直接打开）。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "case_id": r["case_id"],
                "表路径": r["table"],
                "用户提问": r["question"],
                "预期校验结果": r["expected_validation"],
                "预期要点": r["expected_points"],
                "实际校验结果": r["actual_validation"],
                "是否执行成功": r["exec_success"],
                "错误类型": r["error_type"],
                "实际输出摘要": r["actual_output"],
                "耗时(秒)": round(r["elapsed"], 2),
                "是否通过": "",  # 留空，由你人工判定后填
            })


def write_summary(results: list, out_path: str) -> None:
    """生成汇总统计：代码成功率、校验拦截率、平均耗时、失败清单。"""
    total = len(results)
    if total == 0:
        lines = ["总 case 数：0"]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("\n".join(lines))
        return

    exec_success = sum(1 for r in results if r["_exec_success"])
    validation_triggered = sum(1 for r in results if r["_validation_triggered"])
    avg_elapsed = sum(r["elapsed"] for r in results) / total

    # 失败清单：没执行成功的 case
    failed = [r for r in results if not r["_exec_success"]]

    lines = []
    lines.append("=" * 50)
    lines.append("评测汇总")
    lines.append("=" * 50)
    lines.append(f"总 case 数：{total}")
    lines.append(f"代码成功率：{exec_success}/{total} = {exec_success / total:.1%}")
    lines.append(f"校验拦截率：{validation_triggered}/{total} = {validation_triggered / total:.1%}")
    lines.append(f"平均耗时：{avg_elapsed:.2f} 秒")
    lines.append("")
    if failed:
        lines.append(f"失败 case 清单（{len(failed)} 个）：")
        for r in failed:
            lines.append(f"  - {r['case_id']}（{r['error_type'] or '未知'}）")
    else:
        lines.append("失败 case 清单：无")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


def main() -> None:
    # 命令行参数：testcases.json 路径；不带参数则用项目根目录的 testcases.json
    json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "testcases.json")

    if not os.path.exists(json_path):
        print(f"[错误] 找不到测试用例文件：{json_path}")
        print("请先准备好 testcases.json（格式见 testcases.example.json）")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"共 {len(cases)} 个 case，开始评测（逐条调用真实链路）…\n")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] 跑 {case.get('case_id', '?')}：{case.get('用户提问', '')[:30]} …")
        results.append(run_one_case(case))

    out_dir = os.path.join(PROJECT_ROOT, "eval_results")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "results.csv")
    write_results(results, csv_path)

    summary_path = os.path.join(out_dir, "summary.txt")
    write_summary(results, summary_path)

    print(f"\n逐条结果已写入：{csv_path}")
    print(f"汇总统计已写入：{summary_path}")


if __name__ == "__main__":
    main()
