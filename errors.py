"""统一错误处理模块。

把散落各处的 try-except 收口到这里：按异常类型返回中文友好提示，
避免直接抛出 Traceback，也避免各处重复写 if-else 分类。
"""


class AppError(Exception):
    """业务异常基类：message 就是可直接展示给用户的中文提示。"""


class FileFormatError(AppError):
    """文件格式错误。"""


class EncodingError(AppError):
    """文件编码无法识别。"""


class EmptyDataError(AppError):
    """数据为空。"""


class ExecutionError(AppError):
    """代码执行失败。"""


def friendly_message(exc: Exception) -> str:
    """把任意异常转成中文友好提示，不暴露 Traceback。

    优先识别自己的 AppError（其 message 已是中文）；其次按常见内置异常类型分类。
    """
    if isinstance(exc, AppError):
        return str(exc)
    if isinstance(exc, (UnicodeDecodeError, UnicodeError)):
        return "文件编码无法识别，请另存为 UTF-8 或 GBK 编码后重新上传。"
    if isinstance(exc, FileNotFoundError):
        return "找不到文件，请重新上传。"
    if isinstance(exc, KeyError):
        return f"代码执行失败：找不到字段或键 {exc}，请检查字段名是否写错。"
    if isinstance(exc, NameError):
        return f"代码执行失败：使用了未定义的名称 {exc}。"
    if isinstance(exc, AttributeError):
        return f"代码执行失败：属性或方法不存在 {exc}。"
    if isinstance(exc, (ValueError, TypeError)):
        return f"代码执行失败：{exc}"
    return f"操作失败：{exc}"


def execution_message(exc: Exception) -> str:
    """执行失败时，区分「代码问题」与「数据问题」，给出不同引导。"""
    # —— 代码问题：模型生成的代码本身有错 ——
    if isinstance(exc, (SyntaxError, IndentationError)):
        return f"【代码问题】代码语法有误：{exc}。建议换个问法，让模型重新生成代码。"
    if isinstance(exc, NameError):
        return f"【代码问题】代码使用了未定义的名称 {exc}。建议换个问法重试。"
    if isinstance(exc, AttributeError):
        return f"【代码问题】代码调用了不存在的方法/属性 {exc}。建议换个问法重试。"

    # —— 数据问题：数据本身不支持该操作 ——
    if isinstance(exc, ZeroDivisionError):
        return "【数据问题】数据中存在 0 值导致除零，建议先清洗数据或筛除 0 值。"
    if isinstance(exc, KeyError):
        return f"【数据问题】找不到键 {exc}（可能是筛选/分组后该值不存在），建议检查数据。"
    msg = str(exc)
    if isinstance(exc, ValueError) and ("no numeric" in msg or "aggregate" in msg or "empty" in msg):
        return f"【数据问题】{msg}（可能是该列没有可统计的数值），建议检查数据类型。"

    # —— 兜底按代码问题处理 ——
    return f"【代码问题】{friendly_message(exc)}"
