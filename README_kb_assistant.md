# 个人知识库助手（毕业项目）

Agent 学习路线的毕业作品：把前面 12 个任务拼成一个能用的助手。

## 它能干什么

问它踩坑类问题，它去知识库（`knowledge_base.md`，10 篇真实踩坑文档）里检索，
只根据原文回答；知识库里没有的，老实说不知道。还能记住对话，聊长了自动摘要压缩。

## 技能对照表

| 技能 | 来自任务 | 在本项目里的位置 |
| --- | --- | --- |
| RAG 检索 | 任务 7 | `kb_assistant.py` 的 Retriever |
| 对话记忆 + 摘要压缩 | 任务 2 | `main()` 的 history / summary |
| LangGraph 成图 | 任务 8 | retrieve → answer |
| 可观测 | 任务 10 | 每轮打印检索分数 |
| eval 评分 | 任务 4 | `eval_kb.py` |
| 代理失败直连重试 | 任务 9、10 | `ask_llm` / 检索里的 fallback |

## 怎么跑

```powershell
pip install langgraph sentence-transformers requests numpy
python kb_assistant.py   # 问答，知识库在 knowledge_base.md
python eval_kb.py        # 跑 6 道 eval 题，看通过率
```

需要 `DEEPSEEK_API_KEY` 环境变量；embedding 模型是任务 7 那个
`shibing624/text2vec-base-chinese`（已下载过的直接复用）。
