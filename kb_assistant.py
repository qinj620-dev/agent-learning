"""毕业项目：个人知识库助手 —— RAG + 记忆 + LangGraph + 可观测

拼装前面学过的东西：
  RAG（任务 7）：知识库检索，库内问题按原文回答；库外问题改用大模型自己的
  知识回答，但必须注明不是知识库内容（两档模式，不再一律说不知道）
  记忆（任务 2）：对话超长时摘要压缩，不丢上下文
  LangGraph（任务 8）：检索 → 回答画成图
  可观测（任务 10）：每轮打印检索分数
  eval（任务 4）：见 eval_kb.py

知识库就是任务 7 那份 knowledge_base.md（19 篇知识库文档），
和本文件放同一个目录。
长期记忆（2026-11-06 加）：对话历史 + 摘要存进代码旁边的
SQLite（memory.db），启动时恢复，每轮结束写回；本地永久有效，
Render 免费版休眠/重启会丢运行时文件，线上只在实例醒着时有效。
"""
import math
import os
import re
import sqlite3
import threading
import requests
from collections import Counter
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-chat"
API = "https://api.deepseek.com/chat/completions"
KB_PATH = "knowledge_base.md"
SIM_THRESHOLD = 0.1  # TF-IDF 余弦分数：实测库内 0.18–0.31、库外 0.0–0.03，取中间值

RETRIEVER = None  # main() 里初始化，节点通过它查库

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db")


class MemoryStore:
    """长期记忆：对话历史 + 摘要，存进代码旁边的 SQLite（memory.db）。

    为什么用 SQLite：标准库自带，不用装任何东西；一个文件就是全部记忆，
    复制走就能带走；读写毫秒级，几十条对话才几十 KB。

    用法：启动时 load() 恢复 (summary, history)，每轮问答结束 save() 写回。
    本地跑：memory.db 一直在，记忆永久有效。
    Render 免费版：实例休眠/重启会丢运行时文件（2026-11-05 实测：构建时
    没进镜像的模型文件就是这么丢的），所以线上记忆只在实例醒着时有效，
    睡一觉就没了——这是免费档的限制，不是 bug；要永久在线记忆得用云数据库。
    """

    def __init__(self, path=DB_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._broken = False
        try:
            with self._connect() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS meta"
                    " (key TEXT PRIMARY KEY, value TEXT)")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS messages"
                    " (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  role TEXT NOT NULL, content TEXT NOT NULL)")
        except sqlite3.Error as e:
            self._broken = True
            print(f"  [记忆] 数据库打不开（{e}），本次只用内存记忆")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def load(self):
        """读库恢复：返回 (summary, history)，库不存在时返回 ("", [])。"""
        if self._broken:
            return "", []
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT value FROM meta WHERE key='summary'").fetchone()
                summary = row[0] if row else ""
                history = [{"role": r, "content": c} for r, c in conn.execute(
                    "SELECT role, content FROM messages ORDER BY id")]
            return summary, history
        except sqlite3.Error as e:
            print(f"  [记忆] 读库失败（{e}），按空记忆继续")
            return "", []

    def save(self, summary, history):
        """整库写回：删掉旧消息，写入当前 history + summary。
        history 很小（压缩后最多十来条），整存整取最简单可靠。"""
        if self._broken:
            return
        try:
            with self._lock, self._connect() as conn:
                conn.execute("DELETE FROM messages")
                conn.executemany(
                    "INSERT INTO messages (role, content) VALUES (?, ?)",
                    [(m["role"], m["content"]) for m in history])
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value)"
                    " VALUES ('summary', ?)",
                    (summary,))
        except sqlite3.Error as e:
            print(f"  [记忆] 写库失败（{e}），本轮记忆没存上")


def ask_llm(system, user):
    """调 DeepSeek：代理掐 TLS 时改直连重试（任务 9、10 同款坑）"""
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}",
               "Content-Type": "application/json"}
    payload = {"model": MODEL,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    try:
        r = requests.post(API, headers=headers, json=payload, timeout=60)
    except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
        r = requests.post(API, headers=headers, json=payload, timeout=60,
                          proxies={"http": None, "https": None})
    return r.json()["choices"][0]["message"]["content"]


def _tokens(text):
    """分词：英文/数字按词切（pip、SSL 这类词要完整保留），
    中文按字二元组（"报错""推送"这类词不用 jieba 也能对上）。
    不用任何模型或分词库，16 篇小知识库够用。"""
    text = text.lower()
    words = re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", text)
    chars = re.sub(r"\s+", "", text)
    grams = [chars[i:i + 2] for i in range(len(chars) - 1)]
    return words + grams


class Retriever:
    """知识库检索：按 ## 切块 → 纯 Python TF-IDF → 查 top-2。

    不用 torch/embedding 模型：Render 免费版只有 512MB 内存，
    sentence-transformers 一加载就 OOM（2026-11-05 实测，
    Events 明确写着 Ran out of memory used over 512MB）。
    19 篇小知识库，关键词 TF-IDF 足够分清库内/库外。
    """

    def __init__(self, path):
        text = open(path, encoding="utf-8").read()
        self.chunks = [p.strip() for p in text.split("## ") if p.strip()]
        n = len(self.chunks)
        df = Counter()
        tf_list = []
        for c in self.chunks:
            cnt = Counter(_tokens(c))
            tf_list.append(cnt)
            df.update(cnt.keys())
        self._idf = {t: math.log((n + 1) / (d + 1)) + 1
                     for t, d in df.items()}
        self._default_idf = math.log(n + 1) + 1
        # 每篇预先算好归一化 TF-IDF 向量（dict 形式，省内存）
        self._vecs = []
        for cnt in tf_list:
            vec = {t: (1 + math.log(v)) * self._idf[t]
                   for t, v in cnt.items()}
            norm = math.sqrt(sum(x * x for x in vec.values())) or 1.0
            self._vecs.append({t: x / norm for t, x in vec.items()})
        print(f"知识库加载：{len(self.chunks)} 篇")

    def _query_vec(self, query):
        cnt = Counter(_tokens(query))
        vec = {t: (1 + math.log(v)) * self._idf.get(t, self._default_idf)
               for t, v in cnt.items()}
        norm = math.sqrt(sum(x * x for x in vec.values())) or 1.0
        return {t: x / norm for t, x in vec.items()}

    def search(self, query, top_k=2):
        q = self._query_vec(query)
        scored = []
        for i, vec in enumerate(self._vecs):
            s = sum(qv * vec.get(t, 0.0) for t, qv in q.items())
            scored.append((s, i))
        scored.sort(reverse=True)
        return [(self.chunks[i], float(s)) for s, i in scored[:top_k]]


class State(TypedDict):
    question: str
    summary: str       # 旧对话的摘要（记忆压缩的结果）
    history_text: str  # 最近几轮对话
    retrieved: list
    answer: str


def retrieve_node(state: State):
    hits = RETRIEVER.search(state["question"])
    print(f"  [检索] top 分数：{[round(s, 3) for _, s in hits]}")
    return {"retrieved": hits}


def answer_node(state: State):
    hits = state["retrieved"]
    if not hits or hits[0][1] < SIM_THRESHOLD:
        # 库外兜底：不说不知道，改用大模型自己的知识回答，
        # 但必须注明不是知识库内容，让使用者分清来源
        ans = ask_llm(
            "你是一个通用问答助手，请直接回答用户的问题。",
            f"问题：{state['question']}")
        return {"answer": "知识库里没有相关内容，以下是大模型自己的回答，仅供参考：\n" + ans}
    context = "\n\n".join(c for c, _ in hits)
    ans = ask_llm(
        "你是知识库助手。只根据下面资料回答，答不上就说不知道，不要瞎编。",
        f"对话摘要：{state['summary']}\n最近对话：{state['history_text']}\n"
        f"资料：\n{context}\n\n问题：{state['question']}")
    return {"answer": ans}


def build_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answer", answer_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()


def summarize_old(messages):
    """记忆压缩：把较早的对话摘要成 2-3 句话（任务 2 的 summarize）"""
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return ask_llm("把下面对话摘要成 2-3 句话，只保留关键事实。", text)


def main():
    global RETRIEVER
    RETRIEVER = Retriever(KB_PATH)
    app = build_graph()
    store = MemoryStore()
    summary, history = store.load()
    if history or summary:
        print(f"  [记忆] 已恢复上次对话：{len(history)} 条"
              + (f"，摘要：{summary[:60]}..." if summary else ""))
    while True:
        q = input("你问（回车退出）：").strip()
        if not q:
            break
        # 记忆：超 10 条就把旧的摘要压缩，只留最近 4 条
        if len(history) > 10:
            summary = summarize_old(history[:-4])
            history = history[-4:]
            print(f"  [记忆] 已压缩旧对话，摘要：{summary[:60]}...")
        history_text = "\n".join(f"{m['role']}: {m['content']}"
                                 for m in history[-4:])
        result = app.invoke({"question": q, "summary": summary,
                             "history_text": history_text,
                             "retrieved": [], "answer": ""})
        print("\n回答：", result["answer"], "\n")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result["answer"]})
        store.save(summary, history)  # 每轮结束写库，下次启动接着聊


if __name__ == "__main__":
    main()
