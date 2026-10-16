import os
from openai import OpenAI
import json, requests
from datetime import datetime

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
NO_PROXY = {"http": None, "https": None}  # 绕过系统代理，直连

# ---------- 工具 ----------
def get_weather(city: str) -> str:
    """查天气：城市名 -> geocoding 转经纬度 -> Open-Meteo 查预报"""
    city = city.strip()  # 模型传的参数可能带空格，先清洗
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
    """查今天日期：agent 原来没有时间概念，问星期几只能靠猜"""
    week = "一二三四五六日"[datetime.now().weekday()]
    return datetime.now().strftime(f"%Y 年 %m 月 %d 日，星期{week}")

# 工具注册表：告诉模型有哪些工具、干嘛的、参数长什么样
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

# ---------- 记忆管理 ----------
def is_tool_message(m):
    role = m["role"] if isinstance(m, dict) else m.role
    return role == "tool"

def message_to_text(m):
    """把一条消息转成纯文本，方便做总结"""
    if isinstance(m, dict):
        role, content = m["role"], m.get("content") or ""
    else:
        role, content = m.role, m.content or ""
        if getattr(m, "tool_calls", None):
            names = ", ".join(tc.function.name for tc in m.tool_calls)
            content = f"[调用了工具: {names}]"
    return f"{role}: {content}"

def trim_messages(messages, keep=10):
    """策略A：截断 —— 只留 system + 最近 keep 条，不断开工具调用对"""
    if len(messages) <= keep + 1:
        return messages
    start = len(messages) - keep
    while start > 1 and is_tool_message(messages[start]):
        start -= 1
    return [messages[0]] + messages[start:]

def summarize_messages(messages, keep_recent=6):
    """策略B：总结 —— 老消息压缩成摘要，只留最近 keep_recent 条原文"""
    if len(messages) <= keep_recent + 1:
        return messages
    old_text = "\n".join(message_to_text(m) for m in messages[1:-keep_recent])
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "把以下对话历史总结成一段简短中文摘要,保留关键事实(问过哪些城市、用户提过什么偏好)。"},
            {"role": "user", "content": old_text},
        ],
    )
    summary = resp.choices[0].message.content
    start = len(messages) - keep_recent
    while start > 1 and is_tool_message(messages[start]):
        start -= 1  # 别让保留段以 tool 消息开头
    return [
        messages[0],
        {"role": "user", "content": f"[之前对话的摘要]{summary}"},
        *messages[start:],
    ]

# ---------- 主循环 ----------
USE_SUMMARY = True  # 记忆策略开关：True=总结，False=截断

messages = [{"role": "system", "content":
    "你是一个能调用工具的天气助手。工作守则："
    "1. 每次调用工具前，先用一句话说明你要查什么、为什么（格式：[思考] ...）。"
    "2. 拿到工具结果后，先判断信息是否足够回答用户：够了就回答，不够就继续查或向用户追问。"
    "3. 用户没说城市时，绝不瞎猜城市名，必须先追问。"}]
while True:
    q = input("你：")
    if q.strip() in ("quit", "退出"):
        break
    messages.append({"role": "user", "content": q})
    while True:
        try:
            msg = client.chat.completions.create(
                model="deepseek-chat", messages=messages, tools=tools
            ).choices[0].message
        except Exception as e:
            print("助手：调用模型出错了：", e)
            break
        messages.append(msg)
        if msg.tool_calls and msg.content:
            print(f"💭 {msg.content}")  # 反思可视化：看到模型每一步的思考
        if not msg.tool_calls:
            print("助手：" + (msg.content or ""))
            break
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"[工具调用] {tc.function.name}({args})")  # 可观测性
            try:
                if tc.function.name == "get_weather":
                    result = get_weather(**args)
                elif tc.function.name == "get_today":
                    result = get_today()
                else:
                    result = f"未知工具：{tc.function.name}"
            except Exception as e:
                # 工具报错不崩溃，作为结果喂回模型，让它自己决定重试还是澄清
                result = f"工具执行失败：{e}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    # 每轮结束后整理记忆，防止历史无限增长
    if len(messages) > 20:
        if USE_SUMMARY:
            messages = summarize_messages(messages)
            print("(已压缩历史为摘要,当前消息数:", len(messages), ")")
        else:
            messages = trim_messages(messages)
            print("(已截断历史,当前消息数:", len(messages), ")")
input("按回车退出...")



