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


THEME_CSS = """
/* 页面：暖白底、窄栏居中 */
body { background: #fafaf8; }
.gradio-container { max-width: 840px !important; margin: 0 auto; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }

/* 头部 */
.hero { display: flex; gap: 18px; align-items: center; padding: 34px 6px 20px; }
.hero .monogram { width: 68px; height: 68px; min-width: 68px; border-radius: 50%; background: #0f766e; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 32px; font-weight: 600; }
.hero h1 { font-size: 30px; line-height: 1.25; margin: 0 0 6px; color: #1c1917; letter-spacing: 2px; }
.hero .tagline { font-size: 15px; color: #44403c; margin: 0 0 5px; line-height: 1.5; }
.hero .tagline b { color: #0f766e; }
.hero .intro { font-size: 13.5px; color: #78716c; margin: 0; line-height: 1.65; }
.hero .note { font-size: 12.5px; color: #a8a29e; margin: 3px 0 0; line-height: 1.6; }

/* 聊天卡片 */
.chat-card { background: #ffffff; border: 1px solid #e7e5e4 !important; border-radius: 18px !important; box-shadow: 0 10px 32px rgba(28, 25, 23, 0.06); overflow: hidden; margin-bottom: 14px; }

/* 提问行 */
.ask-row { margin-bottom: 14px; gap: 10px; align-items: stretch; }
.send-btn { background: #0f766e !important; color: #ffffff !important; border: none !important; border-radius: 14px !important; min-width: 76px; font-weight: 600; }

/* 提问框 */
.ask-box { background: #ffffff !important; }
.ask-box > div { background: #ffffff !important; }
.ask-box textarea { border-radius: 14px !important; border-color: #d6d3d1 !important; background: #ffffff !important; }

/* 气泡（命中即生效，不命中也不影响） */
.message.user { background: #ccfbf1 !important; }
.message.bot { background: #f5f5f4 !important; }

/* 示例问题：胶囊按钮 */
.examples-wrap { border: none !important; background: transparent !important; box-shadow: none !important; padding: 0 !important; }
.examples-wrap button { border-radius: 999px !important; border: 1px solid #d6d3d1 !important; background: #ffffff !important; color: #44403c !important; }

/* 页脚 */
.footer { text-align: center; font-size: 12.5px; color: #a8a29e; padding: 4px 0 30px; line-height: 1.8; }
.footer a { color: #0f766e; text-decoration: none; }

/* 窄屏：头部纵向排 */
@media (max-width: 640px) {
  .hero { flex-direction: column; align-items: flex-start; gap: 12px; padding-top: 24px; }
  .hero .monogram { width: 56px; height: 56px; min-width: 56px; font-size: 26px; }
  .hero h1 { font-size: 26px; }
}
"""

HERO_HTML = """
<div class="hero">
  <div class="monogram">秦</div>
  <div>
    <h1>秦龙</h1>
    <p class="tagline"><b>Agent 工程师</b> · AI 应用方向 ｜ 上海康琼信息科技有限公司 销售经理</p>
    <p class="intro">问它踩坑类问题，也可以问秦龙的经历、项目和技能。</p>
    <p class="note">答案来自知识库；库里没有的，会用大模型自己的知识回答并注明。</p>
  </div>
</div>
"""

GREETING = "你好，我是秦龙的知识库助手。问我秦龙的经历、项目、技能，或者开发踩坑问题——也可以直接点下面的示例问题试试。"

THEME = gr.themes.Soft(primary_hue="teal", neutral_hue="stone", radius_size="lg")

FOOTER_HTML = """
<div class="footer">
  内容来自秦龙的知识库 · 代码开源在 <a href="https://github.com/qinj620-dev/agent-learning" target="_blank">GitHub</a>
</div>
"""

def build_demo():
    memory = load_memory()
    with gr.Blocks(title="秦龙 · 个人知识库助手") as demo:
        gr.HTML(HERO_HTML)
        initial = ([{"role": m["role"], "content": m["content"]}
                    for m in memory["history"]] if memory["history"] else
                   [{"role": "assistant", "content": GREETING}])
        chatbot = gr.Chatbot(show_label=False, height=480,
                             elem_classes="chat-card",
                             value=initial)
        with gr.Row(elem_classes="ask-row"):
            msg = gr.Textbox(show_label=False, placeholder="比如：秦龙是谁？",
                             elem_classes="ask-box", scale=1)
            send = gr.Button("发送", elem_classes="send-btn", scale=0)
        state = gr.State({"summary": memory["summary"],
                          "history": list(memory["history"])})
        msg.submit(respond, [msg, chatbot, state], [chatbot, state, msg])
        send.click(respond, [msg, chatbot, state], [chatbot, state, msg])
        with gr.Column(elem_classes="examples-wrap"):
            gr.Examples(
                examples=[["秦龙是谁？"], ["秦龙有什么证书？"], ["知识库助手项目用了什么技术？"],
                          ["秦龙的项目经历有哪些？"], ["秦龙的工作经历是怎样的？"], ["pip 安装报错 SSL 怎么解决？"]],
                inputs=msg,
                label="试试这样问",
            )
        gr.HTML(FOOTER_HTML)
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
    demo.launch(css=THEME_CSS, theme=THEME, **launch_kwargs())
