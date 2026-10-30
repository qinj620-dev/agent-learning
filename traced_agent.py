"""任务10：给 Agent 装仪表盘 —— 可观测 + 成本

以前：Agent 跑完只看到最终回答，中间调了几次模型、花了多少 token、
哪步最慢，全凭猜。
现在：每次 LLM 调用、每次工具调用都记一笔，结束打一张汇总表。
这就是可观测（Observability）：出问题能定位，花钱能算账。
"""
import json
import os
import time
import requests

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-chat"
# 单价（元/百万 token），以官网现价为准，变了就改这里
PRICE_IN_PER_M = 1.0
PRICE_OUT_PER_M = 2.0

SYSTEM = "你是一个天气助手。不知道城市时必须追问，不能瞎猜。"

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查询指定城市的实时天气",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "城市名，如 北京"}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_today",
        "description": "获取今天的日期和星期",
        "parameters": {"type": "object", "properties": {}}}},
]


class Tracer:
    """极简追踪器：记事件，结束出汇总"""

    def __init__(self):
        self.events = []

    def log(self, kind, name, detail, latency, in_tokens=0, out_tokens=0, ok=True):
        self.events.append({
            "kind": kind, "name": name, "detail": detail,
            "latency": round(latency, 2),
            "in_tokens": in_tokens, "out_tokens": out_tokens, "ok": ok,
        })
        flag = "OK " if ok else "FAIL"
        print(f"  [trace] {kind:4} {name:12} {latency:5.2f}s {flag} {detail}")

    def report(self):
        llm = [e for e in self.events if e["kind"] == "llm"]
        tool = [e for e in self.events if e["kind"] == "tool"]
        in_tok = sum(e["in_tokens"] for e in llm)
        out_tok = sum(e["out_tokens"] for e in llm)
        cost = in_tok / 1e6 * PRICE_IN_PER_M + out_tok / 1e6 * PRICE_OUT_PER_M
        total_t = sum(e["latency"] for e in self.events)
        print("\n===== 本次运行汇总 =====")
        print(f"LLM 调用：{len(llm)} 次，工具调用：{len(tool)} 次")
        print(f"输入 token：{in_tok}，输出 token：{out_tok}")
        print(f"估算费用：约 ¥{cost:.4f}（单价 {PRICE_IN_PER_M}/{PRICE_OUT_PER_M} 元/百万 token）")
        print(f"总耗时：{total_t:.2f}s")
        slowest = max(self.events, key=lambda e: e["latency"], default=None)
        if slowest:
            print(f"最慢的一步：{slowest['kind']} {slowest['name']}（{slowest['latency']}s）")
        print("========================\n")


def est_tokens(text):
    """token 粗估（API 没给 usage 时兜底）：1 token ≈ 2 个汉字 / 4 个英文字符"""
    if not text:
        return 0
    cn = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cn // 2 + (len(text) - cn) // 4 + 1


_WEATHER_CN = {0: "晴", 1: "大致晴", 2: "多云", 3: "阴",
               45: "雾", 48: "雾凇",
               51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨",
               61: "小雨", 63: "中雨", 65: "大雨",
               71: "小雪", 73: "中雪", 75: "大雪",
               80: "阵雨", 81: "阵雨", 82: "暴雨",
               95: "雷暴", 96: "雷暴伴冰雹", 99: "雷暴伴冰雹"}


def get_weather(city):
    g = _get("https://geocoding-api.open-meteo.com/v1/search",
             {"name": city, "count": 1, "language": "zh", "format": "json"}).json()
    if not g.get("results"):
        return f"找不到城市：{city}"
    r0 = g["results"][0]
    w = _get("https://api.open-meteo.com/v1/forecast",
             {"latitude": r0["latitude"], "longitude": r0["longitude"],
              "current": "temperature_2m,weather_code",
              "daily": "temperature_2m_max,temperature_2m_min",
              "timezone": "auto"}).json()
    cur, daily = w["current"], w["daily"]
    desc = _WEATHER_CN.get(cur["weather_code"], "未知")
    return (f"{r0.get('name', city)}当前 {cur['temperature_2m']}°C，{desc}，"
            f"今日 {daily['temperature_2m_min'][0]}~{daily['temperature_2m_max'][0]}°C")


def get_today():
    from datetime import date
    return date.today().strftime("%Y-%m-%d %A")


TOOLS_IMPL = {"get_weather": get_weather, "get_today": get_today}


def _post(url, headers, payload):
    """POST：先走默认配置，代理掐 TLS 握手时改直连重试（任务 1、任务 9 同款坑）"""
    try:
        return requests.post(url, headers=headers, json=payload, timeout=60)
    except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
        return requests.post(url, headers=headers, json=payload, timeout=60,
                             proxies={"http": None, "https": None})


def _get(url, params):
    """GET：同上，代理失败改直连重试"""
    try:
        return requests.get(url, params=params, timeout=15)
    except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
        return requests.get(url, params=params, timeout=15,
                            proxies={"http": None, "https": None})


def call_llm(messages, tracer):
    """调一次模型，并记一笔账（优先用 API 返回的真实 usage）"""
    t0 = time.time()
    try:
        r = _post("https://api.deepseek.com/chat/completions",
                  {"Authorization": f"Bearer {DEEPSEEK_KEY}",
                   "Content-Type": "application/json"},
                  {"model": MODEL, "messages": messages,
                   "tools": TOOLS, "tool_choice": "auto"}).json()
        msg = r["choices"][0]["message"]
        usage = r.get("usage", {})
        in_tok = usage.get("prompt_tokens") or est_tokens(
            json.dumps(messages, ensure_ascii=False))
        out_tok = usage.get("completion_tokens") or est_tokens(msg.get("content") or "")
        tracer.log("llm", MODEL, f"in={in_tok} out={out_tok}",
                   time.time() - t0, in_tok, out_tok)
        return msg
    except Exception as e:
        tracer.log("llm", MODEL, f"调用失败：{e}", time.time() - t0, ok=False)
        raise


def run(question):
    tracer = Tracer()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    for _ in range(5):
        msg = call_llm(messages, tracer)
        messages.append({"role": "assistant", "content": msg.get("content"),
                         "tool_calls": msg.get("tool_calls")})
        tcs = msg.get("tool_calls") or []
        if not tcs:
            print("\n回答：", msg.get("content"), "\n")
            break
        for tc in tcs:
            fn = tc["function"]
            args = json.loads(fn["arguments"] or "{}")
            t0 = time.time()
            try:
                result = TOOLS_IMPL[fn["name"]](**args)
                tracer.log("tool", fn["name"], json.dumps(args, ensure_ascii=False),
                           time.time() - t0)
            except Exception as e:
                result = f"工具执行失败：{e}"
                tracer.log("tool", fn["name"], str(e), time.time() - t0, ok=False)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    tracer.report()


if __name__ == "__main__":
    while True:
        q = input("你问（回车退出）：").strip()
        if not q:
            break
        run(q)
