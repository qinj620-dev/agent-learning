"""毕业项目扩展：知识库助手网页版（Gradio）

把 kb_assistant.py 变成网页：别人打开链接就能问。
和 kb_assistant.py、knowledge_base.md 放同一个目录。

本地跑：python web_kb_assistant.py
  - 本机地址 http://127.0.0.1:7860
  - share=True 还会给一个公开链接（gradio.live），别人不用装任何东西就能用；
    注意：公开链接是临时的，电脑关机或程序退出就没了；
    想长期公开，下一步可以部署到 Hugging Face Spaces。
部署到 Render 等平台时：平台会设 PORT 环境变量，程序自动绑 0.0.0.0:$PORT
且不开 share（平台自己给固定链接）；本地没 PORT 时保持上面的默认行为。

长期记忆（2026-11-06 加）：对话历史 + 摘要存在代码旁边的 SQLite
（memory.db），和命令行版共用一份记忆；启动时恢复，每轮问答结束写库。
先做单用户全局记忆，简单可靠。本地永久有效；Render 免费版休眠/重启
会丢运行时文件，线上记忆只在实例醒着时有效。
"""
import os
import threading

import gradio as gr

import kb_assistant as kb

GRAPH = None
_STORE = None
_MEMORY = {"summary": "", "history": []}  # 单用户全局记忆
_MEMORY_LOCK = threading.Lock()


def get_graph():
    global GRAPH
    if GRAPH is None:
        kb.RETRIEVER = kb.Retriever(kb.KB_PATH)
        GRAPH = kb.build_graph()
    return GRAPH


def get_store():
    """长期记忆的 SQLite 存取（和命令行版共用一份 memory.db）。"""
    global _STORE
    if _STORE is None:
        _STORE = kb.MemoryStore()
    return _STORE


def load_memory():
    """启动时读库恢复记忆（单用户全局）。"""
    global _MEMORY
    summary, history = get_store().load()
    _MEMORY = {"summary": summary, "history": history}
    if history or summary:
        print(f"  [记忆] 已恢复上次对话：{len(history)} 条")
    return _MEMORY


def respond(message, chatbot, state):
    """一次问答：记忆走全局 _MEMORY（单用户），每轮结束写库。"""
    if not message.strip():
        return chatbot, state, ""
    with _MEMORY_LOCK:  # 单用户串行处理，避免多请求同时改记忆
        if _STORE is None:
            load_memory()
        graph = get_graph()
        history = _MEMORY["history"]
        summary = _MEMORY["summary"]
        # 记忆：超 10 条就把旧的摘要压缩，只留最近 4 条（和 kb_assistant.py 同款）
        if len(history) > 10:
            summary = kb.summarize_old(history[:-4])
            history = history[-4:]
        history_text = "\n".join(f"{m['role']}: {m['content']}"
                                 for m in history[-4:])
        result = graph.invoke({"question": message, "summary": summary,
                               "history_text": history_text,
                               "retrieved": [], "answer": ""})
        answer = result["answer"]
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        _MEMORY["summary"] = summary
        _MEMORY["history"] = history
        get_store().save(summary, history)
        chatbot = chatbot + [{"role": "user", "content": message},
                             {"role": "assistant", "content": answer}]
        state = {"summary": summary, "history": history}
        return chatbot, state, ""


def build_demo():
    memory = load_memory()
    with gr.Blocks(title="个人知识库助手") as demo:
        gr.Markdown("# 个人知识库助手\n问它踩坑类问题，也可以问秦龙的经历、项目和技能；"
                    "答案来自知识库，库里没有的会用大模型自己的知识回答并注明。")
        chatbot = gr.Chatbot(label="问答",
                             value=[{"role": m["role"], "content": m["content"]}
                                    for m in memory["history"]])
        msg = gr.Textbox(label="你问", placeholder="比如：pip SSL 报错怎么办")
        state = gr.State({"summary": memory["summary"],
                          "history": list(memory["history"])})
        msg.submit(respond, [msg, chatbot, state], [chatbot, state, msg])
    return demo


def launch_kwargs():
    """按运行环境决定启动参数。

    平台上（PORT 已设，如 Render）：绑 0.0.0.0:$PORT，不开 share（平台给固定链接）。
    本地（PORT 未设）：默认本机地址 + share=True 给临时公开链接。
    """
    port = os.environ.get("PORT")
    if port:
        return {"server_name": "0.0.0.0", "server_port": int(port), "share": False}
    return {"share": True}


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(**launch_kwargs())
