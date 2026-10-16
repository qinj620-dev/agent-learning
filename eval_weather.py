import os
from openai import OpenAI
import json, time, requests
from datetime import datetime  # 新增：日期工具需要它

# DeepSeek 客户端（OpenAI 兼容接口）
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
NO_PROXY = {"http": None, "https": None}  # 绕过系统代理，直连

# ---------- 工具 ----------
def get_weather(city: str) -> str:
    city = city.strip()  # 新增：清洗输入，模型传的 '北京 ' 带了个空格
    # 先试原名，再试加/去"市"后缀的变体，提高召回
    queries = [city]
    queries.append(city[:-1] if city.endswith("市") else city + "市")
    geo = None
    for q in queries:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
            params={"name": q, "count": 1, "language": "zh"},
            proxies=NO_PROXY, timeout=15).json()
        if r.get("results"):
            geo = r
            break
    if not geo:
        return f"找不到城市：{city}"
    r = geo["results"][0]
    # 防静默替换：返回的地名必须和用户问的对得上
    found = r.get("name", "")
    if city not in found and found not in city:
        return f"找不到城市：{city}（最接近的是{found}，请确认城市名）"
    w = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": r["latitude"], "longitude": r["longitude"],
        "current": "temperature_2m,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "auto"}, proxies=NO_PROXY, timeout=15).json()
    return json.dumps(w, ensure_ascii=False)
def get_today() -> str:
    """新增：查今天日期。agent 原来没有时间概念，问星期几只能靠猜（还猜错过），
    缺什么能力就补什么工具，这是正解。"""
    week = "一二三四五六日"[datetime.now().weekday()]  # weekday() 周一=0
    return datetime.now().strftime(f"%Y 年 %m 月 %d 日，星期{week}")

# 工具注册表：告诉模型有哪些工具、每个是干嘛的、参数长什么样
tools = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查询指定城市的实时天气和未来气温",
        "parameters": {"type": "object",
            "properties": {"city": {"type": "string", "description": "城市名，如 北京"}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_today",
        "description": "获取今天的日期和星期。当用户问时间、日期相关问题时调用，不要猜。",
        "parameters": {"type": "object", "properties": {}}}},
]

def ask(question):
    """跑完一轮 ReAct 对话，返回 (回答, 用过的工具列表, 消耗的 token 数)"""
    messages = [{"role": "system", "content":
             "你是一个能调用工具的天气助手。重要规则：用户询问的城市名如果可能是笔误或不存在，"
             "不要自行替换为相近的城市去查询，必须先向用户澄清确认。"},
            {"role": "user", "content": question}]
    used_tools, total_tokens = [], 0
    while True:
        # temperature=0：eval 时关掉随机性，保证结果可复现
        resp = client.chat.completions.create(model="deepseek-chat", messages=messages, tools=tools, temperature=0)
        total_tokens += resp.usage.total_tokens
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            return (msg.content or ""), used_tools, total_tokens
        for tc in msg.tool_calls:
            used_tools.append(tc.function.name)
            args = json.loads(tc.function.arguments)
            print(f"[工具调用] {tc.function.name}({args})")  # 可观测性：看清模型到底传了什么
            # 按工具名分发：以后加新工具，在这里加一个分支就行
            try:
                if tc.function.name == "get_weather":
                    result = get_weather(**args)
                elif tc.function.name == "get_today":
                    result = get_today()
                else:
                    result = f"未知工具：{tc.function.name}"
            except Exception as e:
                # 工具报错不直接崩，而是作为"工具结果"喂回模型，让它自己决定重试还是澄清
                result = f"工具执行失败：{e}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

def judge(question, answer, used_tools):
    """LLM-as-judge：返回 (分数, 理由)。1-5 分，5=准确诚实，1=答非所问或编造。"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content":
                 "你是一个严格的评分员。根据用户问题和助手回答打分："
                 "5=准确完整地回答了问题，且诚实（不知道就说不知道）；"
                 "3=基本回答了问题，但有小瑕疵；"
                 "1=答非所问，或编造了信息。"
                 "重要：助手可以通过工具（如 get_weather 查天气、get_today 查日期）获取真实数据，"
                 "如果'助手调用的工具'非空，工具返回的数据就是真实的，"
                 "不要因为'未注明数据来源'扣分。只有没调工具却给出具体数据，才算编造。"
                 "返回格式：第一行只写数字，第二行写一句话理由。"},
                {"role": "user", "content":
                 f"问题：{question}\n助手调用的工具：{used_tools or '无'}\n回答：{answer[:600]}"},
            ],
            temperature=0,
        )
        lines = resp.choices[0].message.content.strip().split("\n")
        score = int(lines[0].strip()[0])  # 只取第一行的数字，防模型多说废话
        reason = lines[1].strip() if len(lines) > 1 else "无理由"
        return score, reason
    except Exception:
        return None, None  # 评委自己崩了也不影响主流程，返回空即可

# ---------- 12 道测试题 ----------
# 每个用例：q=问题；contains=回答必须包含的关键词；not_contains=不能出现（防编造）；
# tool=是否应该调用工具。注意测试用例要跟着 agent 的能力一起进化。
TESTS = [
    {"q": "北京明天天气怎么样", "contains": ["北京", "°C"], "tool": True},
    {"q": "上海和杭州明天哪个暖和", "contains": ["上海", "杭州"], "tool": True},
    {"q": "广州现在多少度", "contains": ["广州", "°C"], "tool": True},
    {"q": "北京后天天气怎么样", "contains": ["北京"], "tool": True},
    {"q": "温州市天气怎么样", "contains": ["温州"], "tool": True},
    {"q": "你好", "tool": False},
    {"q": "谢谢", "tool": False},
    {"q": "1+1等于几", "tool": False, "not_contains": ["°C"]},
    # 第 9 题期望变了：现在有了日期工具，应该调工具而不是瞎猜
    {"q": "今天星期几", "tool": True},
    {"q": "兰兰天气怎么样", "contains": ["兰兰"], "not_contains": ["°C"], "tool": False},
    {"q": "阿斯加德天气怎么样", "contains": ["神话"], "tool": False},
    {"q": "明天适合去爬山吗", "tool": False},  # 观察项：没说城市，它会追问还是瞎猜？
]

# ---------- 跑分 ----------
passed, total_tokens_all, judge_scores = 0, 0, []
for i, t in enumerate(TESTS, 1):
    print(f"[{i}/{len(TESTS)}] 问：{t['q']}")
    t0 = time.time()
    try:
        answer, used_tools, tokens = ask(t["q"])
    except Exception as e:
        print(f"  ❌ 运行出错：{e}\n")
        continue
    total_tokens_all += tokens
    # 规则评分：关键词 + 工具调用行为
    checks = []
    if "contains" in t:
        checks.append(all(k in answer for k in t["contains"]))
    if "not_contains" in t:
        checks.append(all(k not in answer for k in t["not_contains"]))
    if "tool" in t:
        checks.append((len(used_tools) > 0) == t["tool"])
    ok = all(checks)
    passed += ok
    # 评委评分：覆盖"答得好不好"这种主观维度
    j, reason = judge(t["q"], answer, used_tools)
    if j:
        judge_scores.append(j)
        print(f"  评委打分：{j}/5（{reason}）")
    print(f"  {'✅' if ok else '❌'} 工具：{used_tools or '无'} token：{tokens} 耗时：{time.time()-t0:.1f}s")
    if not ok:
        print(f"  回答节选：{answer[:80]}...")
    print()
print(f"总分：{passed}/{len(TESTS)}，评委平均分：{sum(judge_scores)/len(judge_scores):.1f}/5，共消耗 token：{total_tokens_all}")
input("跑分结束，按回车退出...")
