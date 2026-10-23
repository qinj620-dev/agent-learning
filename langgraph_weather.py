"""任务8：LangGraph 重写 —— 把 ReAct 的 while 循环画成一张图

原来（手排班）：
    while True:
        模型决策 → 有 tool_call？→ 调工具 → 结果塞回 → 再问模型

现在（画地图）：
    START → agent →（条件边：有 tool_call？）→ tools → agent → … → END
    节点 = 城市（干活的地方），边 = 路，LangGraph 负责开车
"""
import os
import json
import requests
from datetime import datetime
from openai import OpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ---------- 工具（逻辑和原来一样，精简版） ----------
NO_PROXY = {"http": None, "https": None}  # 绕过系统代理，直连


def get_weather(city: str) -> str:
    """查天气：城市名 -> geocoding 转经纬度 -> Open-Meteo 查预报"""
    city = city.strip()
    queries = [city]
    queries.append(city[:-1] if city.endswith("市") else city + "市")  # 加/去"市"多试，提高召回
    geo = None
    for q in queries:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": q, "count": 1, "language": "zh"},
                         proxies=NO_PROXY, timeout=15).json()
        if r.get("results"):
            geo = r["results"][0]
            break
    if not geo:
        return f"找不到城市：{city}"
    lat, lon = geo["latitude"], geo["longitude"]
    w = requests.get("https://api.open-meteo.com/v1/forecast",
                     params={"latitude": lat, "longitude": lon,
                             "current": "temperature_2m,weathercode",
                             "daily": "temperature_2m_max,temperature_2m_min",
                             "timezone": "auto"},
                     proxies=NO_PROXY, timeout=15).json()
    cur = w["current"]["temperature_2m"]
    tmax = w["daily"]["temperature_2m_max"][0]
    tmin = w["daily"]["temperature_2m_min"][0]
    return f"{geo['name']}当前 {cur}°C，今日 {tmin}~{tmax}°C"


def get_today() -> str:
    return datetime.now().strftime("%Y-%m-%d %A")


TOOLS = {"get_weather": get_weather, "get_today": get_today}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "城市名，如北京"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_today",
        "description": "查询今天的日期和星期几",
        "parameters": {"type": "object", "properties": {}}}},
]

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
SYSTEM = "你是天气助手。用户没说城市时必须追问，不许瞎猜。"

# ---------- 图：状态、节点、边 ----------
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 状态：对话消息列表，自动追加


def agent_node(state: State):
    """节点 agent：调模型做决策（原来 while 循环里的第一步）"""
    resp = client.chat.completions.create(
        model="deepseek-chat", temperature=0,
        messages=[{"role": "system", "content": SYSTEM}] + state["messages"],
        tools=TOOL_SCHEMAS)
    m = resp.choices[0].message
    tcs = [{"id": tc.id, "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
           for tc in (m.tool_calls or [])]
    msg = {"role": "assistant", "content": m.content or ""}
    if tcs:
        msg["tool_calls"] = tcs
    names = [tc["function"]["name"] for tc in tcs]
    print(f"[agent] {'调工具：' + '、'.join(names) if names else '直接回答'}")
    return {"messages": [msg]}


def tool_node(state: State):
    """节点 tools：执行工具调用（原来 while 循环里的第二、三步）"""
    last = state["messages"][-1]
    out = []
    for tc in last["tool_calls"]:
        name, args = tc["function"]["name"], json.loads(tc["function"]["arguments"])
        try:
            result = TOOLS[name](**args)
        except Exception as e:
            result = f"工具报错：{e}"
        print(f"[tools] {name}{args} -> {result[:45]}...")
        out.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    return {"messages": out}


def route(state: State):
    """条件边：看 agent 的决策，有 tool_call 就去 tools，没有就结束"""
    last = state["messages"][-1]
    return "tools" if last.get("tool_calls") else END


graph = StateGraph(State)
graph.add_node("agent", agent_node)   # 城市 agent
graph.add_node("tools", tool_node)    # 城市 tools
graph.add_edge(START, "agent")        # 路：起点 -> agent
graph.add_conditional_edges("agent", route, ["tools", END])  # 路：agent 按决策二选一
graph.add_edge("tools", "agent")      # 路：tools 干完回 agent
app = graph.compile()

try:
    print(app.get_graph().draw_ascii())
except ImportError:  # 没装 grandalf 就看文字版
    print("图结构：START → agent →（有 tool_call？）→ tools → agent …；无 tool_call → END")

while True:
    q = input("你问：").strip()
    if not q:
        break
    result = app.invoke({"messages": [{"role": "user", "content": q}]})
    print("\n回答：", result["messages"][-1]["content"], "\n")
