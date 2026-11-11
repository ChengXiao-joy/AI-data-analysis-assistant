# AI 数据分析助手 V1

面向无代码业务人员的轻量化自助数据分析 Web 工具，基于 Streamlit + DeepSeek LLM API，支持表格上传、数据质检、自然语言数据分析，做了幻觉隔离与隐私管控设计。

> ⚠️ **AI 模块（智能问答）需要自行配置 DeepSeek API Key；纯数据质检、预览功能可直接运行（无需 Key）。**

## 环境要求
- Python 3.11+

## 安装依赖
```bash
pip install -r requirements.txt
```

## 配置大模型 API Key（仅「智能问答」功能需要）
1. 复制 `.env.example` 为 `.env`；
2. 在 `.env` 里填入 DeepSeek Key：
   ```
   DEEPSEEK_API_KEY=你的key
   ```

## 运行应用
```bash
streamlit run app.py
```

## 验证模型调用层
```bash
python selfcheck_llm.py
```
