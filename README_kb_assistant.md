# 个人知识库助手（毕业项目）

Agent 学习路线的毕业作品：把前面 12 个任务拼成一个能用的助手。

## 它能干什么

问它踩坑类问题，它去知识库（`knowledge_base.md`，20 篇文档）里检索，
库内问题按原文回答；库外问题改用大模型自己的知识回答，但注明不是知识库内容
（两档模式，不再一律说不知道）。还能记住对话，聊长了自动摘要压缩；
记忆存在代码旁边的 SQLite（`memory.db`），关掉重开接着聊（见下面“长期记忆”）。

## 技能对照表

| 技能 | 来自任务 | 在本项目里的位置 |
| --- | --- | --- |
| RAG 检索 | 任务 7 | `kb_assistant.py` 的 Retriever |
| 对话记忆 + 摘要压缩 | 任务 2 | `MemoryStore`（SQLite `memory.db`）+ 摘要压缩 |
| LangGraph 成图 | 任务 8 | retrieve → answer |
| 可观测 | 任务 10 | 每轮打印检索分数 |
| eval 评分 | 任务 4 | `eval_kb.py` |
| 代理失败直连重试 | 任务 9、10 | `ask_llm` / 检索里的 fallback |

## 长期记忆（SQLite）

对话历史和摘要存在代码旁边的 `memory.db`（Python 标准库 sqlite3，
不用装任何东西）。程序启动时读库恢复记忆，每轮问答结束写回；
关掉重开，接着上次聊。命令行版用这份长期记忆；网页版每会话独立，关页面即消失，不写这份记忆。

- 清空记忆：直接删掉 `memory.db`（下次启动自动重建空库）。
- 本地：永久有效。
- Render 免费版：实例 15 分钟没人访问就休眠，重启会丢运行时文件
所以线上记忆只在实例醒着时有效，睡一觉就没了——这是免费档的限制，不是 bug。
  要永久在线记忆，得用云数据库（下一步）。

## 怎么跑

```powershell
pip install -r requirements.txt
python kb_assistant.py   # 问答，知识库在 knowledge_base.md
python eval_kb.py        # 跑 6 道 eval 题，看通过率
```

需要 `DEEPSEEK_API_KEY` 环境变量；检索是纯 Python 实现的 TF-IDF
（字二元组 + 英文词），不用模型、不用 torch，所以也不用下载任何模型。
