# AI 数据分析助手 V1

面向无代码业务人员的轻量化自助数据分析 Web 工具，基于 Streamlit + DeepSeek LLM API，支持表格上传、数据质检、自然语言数据分析，做了幻觉隔离与隐私管控设计。

> ⚠️ **AI 模块（智能问答）需要 DeepSeek API Key；纯数据质检、预览功能无需 Key 可直接运行。**

## 功能特性

- 📁 **表格上传**：CSV/Excel（≤100MB），自动识别中文编码，支持多 Sheet 与合并分析
- 📄 **数据预览**：前 10 行 + 字段类型标签 + 有效样本/唯一用户/时间范围
- 🔍 **自动质检**：缺失率、重复行、3σ 异常值、类型冲突，风险标红 + 清洗建议（只建议、不改原数据）
- 💬 **自然语言问答**：提问 → 生成 pandas 代码 → 前置校验 → 本地沙盒执行 → 结果分四区展示
- 🛡️ **幻觉隔离**：只上传「字段名 + ≤5 行样本」给大模型，原始数据全程本地；校验层拦截字段不存在/样本过少/逻辑风险
- 📈 **中文图表**：matplotlib 中文标题/坐标轴，按数据特征自动选图型

## 环境要求

- Python 3.11+

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

Key 二选一：
1. 复制 `.env.example` 为 `.env`，填入 `DEEPSEEK_API_KEY`；
2. 或启动后在「智能问答」页展开「🔑 API Key 设置」粘贴 Key。

## Streamlit Cloud 部署

1. 把代码推送到 GitHub 仓库的 **main 分支**；
2. 打开 https://share.streamlit.io ，点 **Create app**，选择本仓库，**Main file path 填 `app.py`**；
3. 在应用后台 **Settings → Secrets** 里配置 DEEPSEEK_API_KEY（见下）；
4. 点击 **Deploy**，完成后即可访问。

> 云端不会读取本地 `.env` 文件，请使用 Secrets 或页面输入框配置 Key。

## Secrets 配置说明

Streamlit Cloud 后台 → 你的应用 → **Settings → Secrets**，粘贴：

```toml
DEEPSEEK_API_KEY = "sk-你的key"
```

**Key 优先级：页面「🔑 API Key 设置」输入框 > Secrets 里的 DEEPSEEK_API_KEY**。都没有时页面会友好提示，不会报错崩溃。

## 使用说明

1. 上传 CSV/Excel（或点「载入示例数据」体验）；
2. 切到「数据质检报告」查看自动质检结果、风险标红与清洗建议；
3. 切到「智能问答分析」用自然语言提问，结果按【数据质检摘要】【文字结论】【图表】【推测与局限说明】四区展示。

## 安全说明

- `.env`、`secrets.toml` 等含密钥文件已被 `.gitignore` 排除，**不会上传仓库**；
- 大模型只接收字段名 + ≤5 行样本，**原始明细数据不离开本地**；
- 生成的 pandas 代码在受限沙盒中执行，禁止读写文件、联网、修改系统。
