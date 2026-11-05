"""毕业项目：个人知识库助手 —— RAG + 记忆 + LangGraph + 可观测

拼装前面学过的东西：
  RAG（任务 7）：知识库检索，库内问题按原文回答；库外问题改用大模型自己的
  知识回答，但必须注明不是知识库内容（两档模式，不再一律说不知道）
  记忆（任务 2）：对话超长时摘要压缩，不丢上下文
  LangGraph（任务 8）：检索 → 回答画成图
  可观测（任务 10）：每轮打印检索分数
  eval（任务 4）：见 eval_kb.py

知识库就是任务 7 那份 knowledge_base.md（10 篇踩坑文档），
和本文件放同一个目录。
"""
import os
import requests
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-chat"
API = "https://api.deepseek.com/chat/completions"
KB_PATH = "knowledge_base.md"
SIM_THRESHOLD = 0.5  # 任务 7 的教训：库外问题最高分才 0.257，低于阈值就说不知道

RETRIEVER = None  # main() 里初始化，节点通过它查库


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


class Retriever:
    """知识库检索：按 ## 切块 → 向量化 → 查 top-2（任务 7 的做法）"""

    def __init__(self, path):
        from sentence_transformers import SentenceTransformer
        text = open(path, encoding="utf-8").read()
        self.chunks = [p.strip() for p in text.split("## ") if p.strip()]
        # 优先读代码旁边的 models/ 文件夹（构建时预下载的模型）；
        # 本地没有就走原来的 Hub 名字（自动下载到缓存）
        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "models", "text2vec-base-chinese")
        model_id = local_dir if os.path.isdir(local_dir) else "shibing624/text2vec-base-chinese"
        self.model = SentenceTransformer(model_id)
        self.embs = self.model.encode(self.chunks, normalize_embeddings=True)
        print(f"知识库加载：{len(self.chunks)} 篇")

    def search(self, query, top_k=2):
        import numpy as np
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embs @ q
        idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in idx]


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
    history = []   # 对话记忆：[{role, content}]
    summary = ""
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


if __name__ == "__main__":
    main()
