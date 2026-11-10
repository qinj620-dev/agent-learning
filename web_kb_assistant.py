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

记忆（2026-11-09 改为每会话独立）：每个访客的对话历史只存在自己
浏览器的 gr.State 里，互不干扰、互不可见；关闭页面即消失，不再写
入共用的 memory.db。命令行版 kb_assistant.py 仍用 SQLite 长期记忆，
本地长期记忆不受影响。
流式回答（2026-11-09 加）：respond 是生成器函数，边生成边把气泡
内容推给前端，不用等整段答完。
清空对话（2026-11-09 加）：点“清空对话”回到问候语，会话记忆清零。
答案出处（2026-11-09 加）：库内命中时答案末尾注明来源篇名，只显示
不写入记忆，上下文里不带这行。
示例直发（2026-11-09 加）：点示例问题直接提问，不用再按发送。
等待与出错（2026-11-09 加）：出首字前气泡显示“思考中…”；模型
连不上或没吐字时给友好提示，并把问题放回提问框，点发送就能重试，
失败的那轮不进记忆。
"""
import os
import threading

import gradio as gr

import kb_assistant as kb

GRAPH = None
_GRAPH_LOCK = threading.Lock()

THINKING = "思考中…"
ERROR_TEXT = "抱歉，刚才没连上模型，可能是网络波动，问题已经放回提问框，再点一次发送试试。"
EMPTY_TEXT = "这次没等到模型的回答，请再试一次。"


def get_graph():
    """建图并初始化检索器（只建一次，多会话共用，只读无冲突）。"""
    global GRAPH
    if GRAPH is None:
        with _GRAPH_LOCK:
            if GRAPH is None:
                kb.RETRIEVER = kb.Retriever(kb.KB_PATH)
                GRAPH = kb.build_graph()
    return GRAPH


def respond(message, chatbot, state):
    """一次问答（生成器）：记忆只走本会话的 state，答案流式吐出。

    history 只在整段答案生成完后才追加，流式中途出错不留半截回答；
    出错时把问题放回提问框，点发送就能重试。
    """
    if not message.strip():
        yield chatbot, state, ""
        return
    get_graph()
    history = state["history"]
    summary = state["summary"]
    # 记忆：超 10 条就把旧的摘要压缩，只留最近 4 条（和 kb_assistant.py 同款）；
    # 摘要失败不影响本轮提问，沿用旧摘要继续
    if len(history) > 10:
        try:
            summary = kb.summarize_old(history[:-4])
        except Exception as e:
            print(f"  [记忆] 摘要压缩失败（{e}），沿用旧摘要")
        history = history[-4:]
    history_text = "\n".join(f"{m['role']}: {m['content']}"
                             for m in history[-4:])
    hits = kb.RETRIEVER.search(message)
    print(f"  [检索] top 分数：{[round(s, 3) for _, s in hits]}")
    system, user_prompt, prefix, source = kb.build_answer_prompts(
        message, summary, history_text, hits)
    chatbot = chatbot + [{"role": "user", "content": message},
                         {"role": "assistant", "content": prefix + THINKING}]
    yield chatbot, state, ""
    full = prefix
    started = False
    try:
        for piece in kb.ask_llm_stream(system, user_prompt):
            if not started:
                full = prefix + piece
                started = True
            else:
                full += piece
            chatbot[-1] = {"role": "assistant", "content": full}
            yield chatbot, state, ""
    except Exception as e:
        print(f"  [回答] 流式调用失败（{e}）")
        chatbot[-1] = {"role": "assistant", "content": ERROR_TEXT}
        yield chatbot, state, message
        return
    if not started:
        # 模型一个字都没吐：不记入历史，问题放回提问框
        chatbot[-1] = {"role": "assistant", "content": EMPTY_TEXT}
        yield chatbot, state, message
        return
    display = full + (f"\n\n*来源：{source}*" if source else "")
    chatbot[-1] = {"role": "assistant", "content": display}
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": full})
    state = {"summary": summary, "history": history}
    yield chatbot, state, ""


def clear_chat():
    """清空对话：回到问候语，本会话记忆清零。"""
    return [{"role": "assistant", "content": GREETING}], \
        {"summary": "", "history": []}, ""


EXAMPLE_QUESTIONS = ["秦龙是谁？", "秦龙有什么证书？", "知识库助手项目用了什么技术？",
                     "秦龙的项目经历有哪些？", "秦龙的工作经历是怎样的？", "pip 安装报错 SSL 怎么解决？"]


def make_ask_example(question):
    """示例按钮点即发送：把示例问题直接当作一次提问。"""
    def _ask(chatbot, state):
        yield from respond(question, chatbot, state)
    return _ask


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
.clear-btn { background: #ffffff !important; color: #57534e !important; border: 1px solid #d6d3d1 !important; border-radius: 14px !important; min-width: 76px; font-weight: 600; }

/* 提问框 */
.ask-box { background: #ffffff !important; }
.ask-box > div { background: #ffffff !important; }
.ask-box textarea { border-radius: 14px !important; border-color: #d6d3d1 !important; background: #ffffff !important; }

/* 气泡（命中即生效，不命中也不影响） */
.message.user { background: #ccfbf1 !important; }
.message.bot { background: #f5f5f4 !important; }

/* 示例问题：胶囊按钮，点即发送 */
.examples-label { font-size: 13px !important; color: #a8a29e !important; margin: 2px 0 8px !important; }
.examples-label p { margin: 0 !important; }
.examples-wrap { border: none !important; background: transparent !important; box-shadow: none !important; padding: 0 !important; display: flex !important; flex-direction: row !important; flex-wrap: wrap !important; gap: 8px !important; }
.examples-wrap button { border-radius: 999px !important; border: 1px solid #d6d3d1 !important; background: #ffffff !important; color: #44403c !important; flex: 0 0 auto !important; }

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
    <p class="tagline"><b>售前 / 解决方案 / IT 产品销售</b> · AI 售前 ｜ 上海康琼信息科技有限公司 销售经理</p>
    <p class="intro">问秦龙的经历、项目、技能和求职，也可以问它踩坑类问题。</p>
    <p class="note">答案来自知识库；库里没有的，会用大模型自己的知识回答并注明。</p>
  </div>
</div>
"""

GREETING = "你好，我是秦龙的知识库助手。问我秦龙的经历、项目、技能和求职，或者开发踩坑问题——也可以直接点下面的示例问题试试。"

THEME = gr.themes.Soft(primary_hue="teal", neutral_hue="stone", radius_size="lg")

FOOTER_HTML = """
<div class="footer">
  内容来自秦龙的知识库 · 代码开源在 <a href="https://github.com/qinj620-dev/agent-learning" target="_blank">GitHub</a>
</div>
"""

def build_demo():
    with gr.Blocks(title="秦龙 · 个人知识库助手") as demo:
        gr.HTML(HERO_HTML)
        chatbot = gr.Chatbot(show_label=False, height=480,
                             elem_classes="chat-card",
                             value=[{"role": "assistant", "content": GREETING}])
        with gr.Row(elem_classes="ask-row"):
            msg = gr.Textbox(show_label=False, placeholder="比如：秦龙是谁？",
                             elem_classes="ask-box", scale=1)
            send = gr.Button("发送", elem_classes="send-btn", scale=0)
            clear = gr.Button("清空对话", elem_classes="clear-btn", scale=0)
        # 每会话独立记忆：gr.State 每个访客一份，互不干扰
        state = gr.State({"summary": "", "history": []})
        msg.submit(respond, [msg, chatbot, state], [chatbot, state, msg])
        send.click(respond, [msg, chatbot, state], [chatbot, state, msg])
        clear.click(clear_chat, [], [chatbot, state, msg])
        gr.Markdown("试试这样问", elem_classes="examples-label")
        with gr.Column(elem_classes="examples-wrap"):
            for q in EXAMPLE_QUESTIONS:
                btn = gr.Button(q, scale=0)
                btn.click(make_ask_example(q), [chatbot, state],
                          [chatbot, state, msg])
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
