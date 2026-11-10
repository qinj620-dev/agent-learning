# agent-learning

从 2026 年 10 月开始，用"动手做项目"的方式学 Agent 开发。13 个任务全部完成（2026-11-05），毕业项目已部署上线。

## 🎓 毕业项目：个人知识库助手（RAG 问答系统）

**在线体验：https://agent-learning-l13m.onrender.com**

把简历做成可问答的知识库：问经历、项目、技能和求职，它去 20 篇文档里检索回答；问开发踩坑，它按踩坑笔记回答。

- **两档回答**：检索分达标按知识库原文答；不达标改用大模型自己的知识答，并注明"不是知识库内容"。
- **对话记忆**：LangGraph 管理对话，历史超 10 条自动摘要压缩、仅保留最近 4 条原文；命令行版用 SQLite 长期记忆（`memory.db`），网页版每会话独立。
- **检索**：自研纯 Python TF-IDF（字二元组 + 英文词），零新依赖，不用 torch、不下模型；分流阈值 0.1（实测库内 0.179–0.306 / 库外 0–0.034）。
- **部署**：Render 免费档 + GitHub 自动部署；峰值内存 178MB（加长期记忆后 179MB）；15 分钟无访问休眠，访问后约 1 分钟唤醒。
- **评测**：`eval_kb.py` 6/6 全部通过。

## 🗺️ 13 个任务：全部完成

| # | 任务 | 成果 | 代码 |
|---|---|---|---|
| 1 | 工具调用 | ReAct 天气助手：工具调用 + 异常处理 | `weather_agent.py` |
| 2 | 记忆 | 对话记忆 + 摘要压缩（后由毕业项目的 `MemoryStore` 承载） | `weather_agent.py` |
| 3 | 规划 | Plan-and-Execute：规划 → 确认 → 执行 → 汇总；ReAct vs Plan-and-Execute 取舍复盘 | `Plan_agent.py` |
| 4 | Evals | 规则评分 12/12，LLM-as-judge 平均 5.0/5；亲手抓出"兰兰→兰州静默替换"、"星期四幻觉"等真实 bug | `eval_weather.py` |
| 5 | Reflection | 反思守则 prompt + 思考可视化；基线 vs 改后对比 | `weather_agent.py` |
| 6 | Embedding | 意思越像，向量离得越近 | `embedding_demo.py` |
| 7 | RAG | 给 Agent 接本地知识库；验收：pip SSL 命中正确资料块，库外问题老实说不知道 | `rag_agent.py` |
| 8 | LangGraph | 把 ReAct 的 while 循环画成一张图：state / node / edge 各司其职 | `langgraph_weather.py` |
| 9 | MCP | Client + Server；Open-Meteo 免 key，代理失败自动直连重试 | `mcp_client_demo.py`、`mcp_weather_server.py` |
| 10 | 可观测 | 仪表盘：调用次数、token、耗时、成本（实测一轮：LLM 2 次、工具 1 次，718 / 81 token，约 ¥0.0009，5.89s） | `traced_agent.py` |
| 11 | 多智能体 | 旅行规划小队：天气员 / 交通员 / 景点员并行，汇总员合成行程 | `multi_agent_travel.py` |
| 12 | 安全攻防 | 提示词注入攻防；核心机制：指令和数据分开 | `injection_demo.py` |
| 13 | 毕业项目 | 个人知识库助手：RAG + 记忆 + LangGraph + 可观测 + eval | `kb_assistant.py`、`web_kb_assistant.py`、`app.py`、`eval_kb.py` |

## 📁 仓库里还有什么

- `knowledge_base.md`：知识库正文（20 篇文档）
- `README_kb_assistant.md`：毕业项目详细介绍
- `download_model.py`：已废弃（早期预下载 embedding 模型，现为空操作）
- `requirements.txt`：gradio、langgraph、requests、numpy

## 🚀 本地运行

```bash
pip install -r requirements.txt

# 设置 DeepSeek API key（不要写进代码）
# Windows PowerShell：setx DEEPSEEK_API_KEY "你的key"
# macOS / Linux：export DEEPSEEK_API_KEY="你的key"

python kb_assistant.py        # 命令行问答
python web_kb_assistant.py    # 网页版（Gradio）
python eval_kb.py             # 6 道 eval，看通过率
```

早期练习文件（`weather_agent.py`、`Plan_agent.py` 等）用 OpenAI 兼容接口，需要另外装 `openai` 包：`pip install openai`。

## 👤 关于作者

秦龙，上海康琼信息科技有限公司销售经理。求职方向：售前 / 解决方案 / IT 产品销售 · AI 售前；长期目标是成为一名 Agent 工程师。
