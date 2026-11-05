"""毕业项目 eval：6 道题，测 RAG 的两条底线（任务 4 的规则评分）
  库内问题：必须答对关键点，不能说不知道
  库外问题：必须注明不是知识库内容（两档模式，不再要求说不知道）
"""
import kb_assistant as kb

# (问题, 预期关键词, 类型) —— 类型 in 表示库内，out 表示库外
TESTS = [
    ("pip 安装时报 SSL 错误怎么办", "no_proxy", "in"),
    ("git push 被拒绝怎么办", "pull", "in"),
    ("Python 缩进报错怎么解决", "缩进", "in"),
    ("今天北京天气怎么样", "不知道", "out"),
    ("怎么做红烧肉", "不知道", "out"),
    ("明天股票涨还是跌", "不知道", "out"),
]


def run_eval():
    kb.RETRIEVER = kb.Retriever(kb.KB_PATH)
    app = kb.build_graph()
    passed = 0
    for question, keyword, kind in TESTS:
        result = app.invoke({"question": question, "summary": "",
                             "history_text": "", "retrieved": [], "answer": ""})
        ans = result["answer"]
        if kind == "in":
            ok = keyword in ans and "不知道" not in ans
        else:
            ok = "大模型自己的回答" in ans
        passed += ok
        print(f"{'通过' if ok else '失败'} [{kind}] {question} -> {ans[:50]}...")
    print(f"\n通过率：{passed}/{len(TESTS)}")


if __name__ == "__main__":
    run_eval()
