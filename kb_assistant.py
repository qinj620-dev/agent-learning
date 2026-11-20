"""毕业项目：个人知识库助手 —— RAG + 记忆 + LangGraph + 可观测

拼装前面学过的东西：
  RAG（任务 7）：知识库检索，库内问题按原文回答；库外问题先联网搜索
  （Tavily，见 web_search.py），搜到按搜索结果答并注明来源，搜不到才用
  大模型自己的知识回答，并注明不是知识库内容（三档模式，不再一律说不知道）
  记忆（任务 2）：对话超长时摘要压缩，不丢上下文
  LangGraph（任务 8）：检索 → 回答画成图
  可观测（任务 10）：每轮打印检索分数
  eval（任务 4）：见 eval_kb.py

知识库就是任务 7 那份 knowledge_base.md（19 篇知识库文档），
和本文件放同一个目录。
长期记忆（2026-11-06 加）：对话历史 + 摘要存进代码旁边的
SQLite（memory.db），启动时恢复，每轮结束写回；本地永久有效，
Render 免费版休眠/重启会丢运行时文件，线上只在实例醒着时有效。
流式回答（2026-11-09 加）：ask_llm_stream 供网页版边生成边显示，
命令行版和 eval 仍走一次性 ask_llm，行为不变。
答案出处（2026-11-09 加）：build_answer_prompts 顺带返回命中篇名，
供网页版在答案后注明来源。
联网搜索（2026-11-11 加）：库外问题先调 Tavily 搜一圈，搜到按搜索结果答
并注明来源网址，搜不到或没配 TAVILY_API_KEY 时回落到大模型知识；没 key 不
报错、不影响原有两档。见 web_search.py。
云数据库（2026-11-20 加）：MemoryStore 两档存储，设了 TURSO_DATABASE_URL +
TURSO_AUTH_TOKEN 走 Turso 云数据库（libSQL），Render 重启也不丢记忆；没设
走本地 SQLite，和以前一样。
"""
import sys
import json
import math
import os
import re
import sqlite3
import threading
import requests
import web_search
from collections import Counter
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
TURSO_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")
MODEL = "deepseek-chat"
API = "https://api.deepseek.com/chat/completions"
KB_PATH = "knowledge_base.md"
SIM_THRESHOLD = 0.1  # TF-IDF 余弦分数：实测库内 0.18–0.31、库外 0.0–0.03，取中间值

RETRIEVER = None  # main() 里初始化，节点通过它查库

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db")


class MemoryStore:
    """长期记忆：对话历史 + 摘要。

    两档存储（2026-11-20 加）：
    - 设了 TURSO_DATABASE_URL + TURSO_AUTH_TOKEN：走 Turso 云数据库（libSQL，
      和 SQLite 同一个方言），Render 重启/休眠也不丢记忆。
    - 没设：走本地 SQLite（memory.db），和以前一样。
    两边表结构、SQL 完全一样，只是连接方式不同，所以只抽象出一个 _run()
    统一执行，load/save 不用分两套写。
    """

    def __init__(self, path=DB_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._broken = False
        self._remote = bool(TURSO_URL and TURSO_TOKEN)
        self._client = None
        try:
            if self._remote:
                import libsql_client
                self._client = libsql_client.create_client_sync(
                    url=TURSO_URL, auth_token=TURSO_TOKEN)
            self._run("CREATE TABLE IF NOT EXISTS meta"
                      " (key TEXT PRIMARY KEY, value TEXT)")
            self._run("CREATE TABLE IF NOT EXISTS messages"
                      " (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                      "  role TEXT NOT NULL, content TEXT NOT NULL)")
        except Exception as e:
            self._broken = True
            print(f"  [记忆] 数据库打不开（{e}），本次只用内存记忆")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def _run(self, sql, params=None, fetch=False):
        """在当前后端上执行一条 SQL；fetch=True 时返回行列表（每行是 tuple）。"""
        if self._remote:
            rs = self._client.execute(sql, list(params) if params else [])
            return [tuple(r) for r in rs.rows] if fetch else None
        with self._connect() as conn:
            cur = conn.execute(sql, params or ())
            return cur.fetchall() if fetch else None

    def close(self):
        """关闭远程连接（本地 SQLite 不用关）。
        libsql 同步客户端背后有个事件循环线程，不关它进程退不出。"""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def load(self):
        """读库恢复：返回 (summary, history)，库不存在时返回 ("", [])。"""
        if self._broken:
            return "", []
        try:
            with self._lock:
                rows = self._run(
                    "SELECT value FROM meta WHERE key='summary'", fetch=True)
                summary = rows[0][0] if rows else ""
                history = [{"role": r, "content": c} for r, c in self._run(
                    "SELECT role, content FROM messages ORDER BY id",
                    fetch=True)]
            return summary, history
        except Exception as e:
            print(f"  [记忆] 读库失败（{e}），按空记忆继续")
            return "", []

    def save(self, summary, history):
        """整库写回：删掉旧消息，写入当前 history + summary。
        history 很小（压缩后最多十来条），整存整取最简单可靠。"""
        if self._broken:
            return
        try:
            with self._lock:
                self._run("DELETE FROM messages")
                for m in history:
                    self._run(
                        "INSERT INTO messages (role, content) VALUES (?, ?)",
                        (m["role"], m["content"]))
                self._run(
                    "INSERT OR REPLACE INTO meta (key, value)"
                    " VALUES ('summary', ?)",
                    (summary,))
        except Exception as e:
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


def ask_llm_stream(system, user):
    """调 DeepSeek（流式）：逐片吐出正文，供网页版边生成边显示。

    同 ask_llm 的代理重试逻辑；SSE 每行形如
    data: {"choices": [{"delta": {"content": "..."}}]}，以 data: [DONE] 结束。
    """
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}",
               "Content-Type": "application/json"}
    payload = {"model": MODEL, "stream": True,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    try:
        r = requests.post(API, headers=headers, json=payload, timeout=60,
                          stream=True)
    except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
        r = requests.post(API, headers=headers, json=payload, timeout=60,
                          stream=True, proxies={"http": None, "https": None})
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            piece = json.loads(data)["choices"][0]["delta"].get("content", "")
        except (KeyError, IndexError, ValueError):
            continue
        if piece:
            yield piece


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


def chunk_title(chunk):
    """取知识块的篇名：首行，去掉 # 号与空白。

    切块按 "## " 切，篇名就是每块的首行；过长或取不到时返回空字符串，
    调用方拿不到篇名就不显示来源。
    """
    first = chunk.strip().split("\n", 1)[0].strip().lstrip("#").strip()
    if not first or len(first) > 25:
        return ""
    return first


def fetch_web_if_missed(question, retrieved):
    """知识库没命中时才联网搜；命中了直接返回 []，不浪费搜索次数。"""
    if retrieved and retrieved[0][1] >= SIM_THRESHOLD:
        return []
    return web_search.search_web(question)


def build_answer_prompts(question, summary, history_text, retrieved,
                         web_results=None):
    """按检索结果拼回答用的提示词（一次性与流式共用）。

    三档：库内命中按知识库答；没命中但联网搜到了，按搜索结果答并注明
    来源网址；都没搜到才用大模型自己的知识回答，并注明不是知识库内容。
    返回 (system, user, prefix, source)：prefix 是库外时贴在答案前的
    注明，库内命中时为空字符串；source 是命中篇名或搜索结果网址，库外
    且无搜索结果时为空。
    """
    hits = retrieved
    if not hits or hits[0][1] < SIM_THRESHOLD:
        if web_results:
            # 第三档：联网搜到了，按搜索结果答，来源写第一条结果的网址
            context = "\n\n".join(
                f"{w.get('title', '')}\n{w.get('snippet', '')}\n来源：{w.get('url', '')}"
                for w in web_results)
            return ("你是知识库助手。请只根据下面联网搜索到的资料回答，"
                    "答不上就说不知道，不要瞎编。回答要完整，把资料里的"
                    "关键事实都提到，不要只回一句话。",
                    f"问题：{question}\n\n联网搜索资料：\n{context}",
                    "知识库里没有相关内容，以下是联网搜索到的结果，仅供参考：\n",
                    web_results[0].get("url", "") or "联网搜索")
        # 库外兜底：不说不知道，改用大模型自己的知识回答，
        # 但必须注明不是知识库内容，让使用者分清来源
        return ("你是一个通用问答助手，请直接回答用户的问题。",
                f"问题：{question}",
                "知识库里没有相关内容，以下是大模型自己的回答，仅供参考：\n",
                "")
    context = "\n\n".join(c for c, _ in hits)
    return ("你是知识库助手。只根据下面资料回答，答不上就说不知道，不要瞎编。"
            "回答要完整，把资料里的关键事实都提到，不要只回一句话。",
            f"对话摘要：{summary}\n最近对话：{history_text}\n"
            f"资料：\n{context}\n\n问题：{question}",
            "",
            chunk_title(hits[0][0]))


def answer_node(state: State):
    web_results = fetch_web_if_missed(state["question"], state["retrieved"])
    system, user, prefix, _source = build_answer_prompts(
        state["question"], state["summary"], state["history_text"],
        state["retrieved"], web_results=web_results)
    return {"answer": prefix + ask_llm(system, user)}


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
    try:
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
    except KeyboardInterrupt:
        print("\n再见！")
    # 退出：对话已逐轮写库，这里直接强制结束进程。
    # 原因：libsql 同步客户端起了一个后台线程，正常退出时它有时关不掉，
    # 会拖住进程不退出（表现为按回车没反应）。os._exit() 不走 atexit、
    # 不等后台线程，直接结束，数据已保存、不受影响。
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
