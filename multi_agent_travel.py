"""任务11：多智能体小队 —— 旅行规划

以前：一个 Agent 单打独斗，又要查天气、又要想交通、又要排景点。
现在：组个 4 人小队，各管一摊，最后汇总。

    START ─┬─→ 天气员 ─┐
           ├─→ 交通员 ─┼─→ 汇总员 ─→ END
           └─→ 景点员 ─┘

三个队员并行开工（LangGraph 自动等齐），汇总员看黑板出最终行程。
分工 = 每个节点自己的角色 prompt；协作 = 共享 state 这块黑板；
汇总 = summary 节点。
"""
import os
import requests
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-chat"
API = "https://api.deepseek.com/chat/completions"


class State(TypedDict):
    city: str
    days: int
    weather: str
    transport: str
    spots: str
    plan: str


def _post(payload):
    """调 DeepSeek：代理掐 TLS 时改直连重试（任务 9、10 同款坑）"""
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}",
               "Content-Type": "application/json"}
    try:
        return requests.post(API, headers=headers, json=payload, timeout=60)
    except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
        return requests.post(API, headers=headers, json=payload, timeout=60,
                             proxies={"http": None, "https": None})


def ask_llm(system, user):
    r = _post({"model": MODEL,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}).json()
    return r["choices"][0]["message"]["content"]


def _get(url, params):
    """GET：代理失败改直连重试"""
    try:
        return requests.get(url, params=params, timeout=15)
    except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
        return requests.get(url, params=params, timeout=15,
                            proxies={"http": None, "https": None})


_WEATHER_CN = {0: "晴", 1: "大致晴", 2: "多云", 3: "阴",
               45: "雾", 48: "雾凇",
               51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨",
               61: "小雨", 63: "中雨", 65: "大雨",
               71: "小雪", 73: "中雪", 75: "大雪",
               80: "阵雨", 81: "阵雨", 82: "暴雨",
               95: "雷暴", 96: "雷暴伴冰雹", 99: "雷暴伴冰雹"}


def weather_agent(state: State):
    """天气员：查真实天气（Open-Meteo，免 key）"""
    city = state["city"]
    g = _get("https://geocoding-api.open-meteo.com/v1/search",
             {"name": city, "count": 1, "language": "zh", "format": "json"}).json()
    if not g.get("results"):
        return {"weather": f"找不到城市：{city}"}
    r0 = g["results"][0]
    w = _get("https://api.open-meteo.com/v1/forecast",
             {"latitude": r0["latitude"], "longitude": r0["longitude"],
              "current": "temperature_2m,weather_code",
              "daily": "temperature_2m_max,temperature_2m_min",
              "timezone": "auto"}).json()
    cur, daily = w["current"], w["daily"]
    desc = _WEATHER_CN.get(cur["weather_code"], "未知")
    out = (f"{r0.get('name', city)}当前 {cur['temperature_2m']}°C，{desc}，"
           f"今日 {daily['temperature_2m_min'][0]}~{daily['temperature_2m_max'][0]}°C")
    print(f"[天气员] {out}")
    return {"weather": out}


def transport_agent(state: State):
    """交通员：给出城际 + 市内交通建议（演示用模型知识，生产应接搜索/MCP 工具）"""
    out = ask_llm(
        "你是一名交通规划员，只给交通建议，不谈景点和天气，3-5 条，要具体。",
        f"用户去{state['city']}玩 {state['days']} 天，从上海出发，给交通建议。")
    print(f"[交通员] {out[:80]}...")
    return {"transport": out}


def spots_agent(state: State):
    """景点员：按天排景点（演示用模型知识，生产应接搜索/MCP 工具）"""
    out = ask_llm(
        "你是一名景点规划员，只排景点行程，不谈交通和天气，按天分段，要具体。",
        f"用户去{state['city']}玩 {state['days']} 天，排一份景点行程。")
    print(f"[景点员] {out[:80]}...")
    return {"spots": out}


def summary_agent(state: State):
    """汇总员：看黑板上的三份产出，合成一份完整行程"""
    out = ask_llm(
        "你是一名旅行汇总员，把天气、交通、景点三份材料合成一份完整行程，"
        "结构清晰，先给一句话总览，再按天展开，最后加一句出行提醒。",
        f"城市：{state['city']}，天数：{state['days']} 天。\n"
        f"天气：{state['weather']}\n交通：{state['transport']}\n景点：{state['spots']}")
    return {"plan": out}


def build_graph():
    g = StateGraph(State)
    g.add_node("weather", weather_agent)
    g.add_node("transport", transport_agent)
    g.add_node("spots", spots_agent)
    g.add_node("summary", summary_agent)
    # 三个队员并行开工
    g.add_edge(START, "weather")
    g.add_edge(START, "transport")
    g.add_edge(START, "spots")
    # 都到齐了再汇总
    g.add_edge("weather", "summary")
    g.add_edge("transport", "summary")
    g.add_edge("spots", "summary")
    g.add_edge("summary", END)
    return g.compile()


if __name__ == "__main__":
    import re
    app = build_graph()
    city = input("去哪个城市：").strip() or "北京"
    days_raw = input("玩几天：").strip() or "3"
    # 容忍 "3天" 这种输入，只取里面的数字
    m = re.search(r"\d+", days_raw)
    days = int(m.group()) if m else 3
    result = app.invoke({"city": city, "days": days})
    print("\n===== 最终行程 =====")
    print(result["plan"])
