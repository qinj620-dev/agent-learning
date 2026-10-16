import os
from openai import OpenAI
import json, re, requests

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
NO_PROXY = {"http": None, "https": None}

def get_weather(city: str) -> str:
    geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "zh"},
        proxies=NO_PROXY, timeout=15).json()
    if not geo.get("results"):
        return f"找不到城市：{city}"
    r = geo["results"][0]
    w = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": r["latitude"], "longitude": r["longitude"],
        "current": "temperature_2m,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "auto"}, proxies=NO_PROXY, timeout=15).json()
    return json.dumps(w, ensure_ascii=False)

def extract_json(text):
    m = re.search(r"\[.*\]", text, re.S)
    return json.loads(m.group(0))

def make_plan(question: str):
    """第1步: 规划 — 让模型把问题拆成步骤"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content":
             "你是一个任务规划师。把用户的问题拆解成若干个查天气的步骤。"
             "只返回 JSON 数组,每个元素形如 {\"city\": \"杭州\", \"why\": \"一句话说明这步的目的\"}。"
             "不要返回任何其他文字。"},
            {"role": "user", "content": question},
        ],
    )
    return extract_json(resp.choices[0].message.content)

def execute_plan(plan):
    """第2步: 执行 — 按计划一步步调工具"""
    results = []
    for i, step in enumerate(plan, 1):
        print(f"[执行 {i}/{len(plan)}] 查{step['city']}天气 —— {step.get('why','')}")
        try:
            data = get_weather(step["city"])
        except Exception as e:
            data = f"查询失败：{e}"
        results.append(f"步骤{i}（{step['city']}）：\n{data}")
    return results

def synthesize(question, plan, results):
    """第3步: 汇总 — 把各步结果综合成最终回答"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是善于综合信息的助手。根据各步骤的执行结果,完整回答用户的问题,给出明确结论和理由。"},
            {"role": "user", "content":
             f"用户问题:{question}\n\n执行计划:{json.dumps(plan, ensure_ascii=False)}\n\n"
             f"各步骤结果:\n" + "\n\n".join(results)},
        ],
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    q = input("你想规划什么复杂问题：")
    print("\n[规划中...]")
    plan = make_plan(q)
    print("计划：")
    for i, s in enumerate(plan, 1):
        print(f"  {i}. 查{s['city']} —— {s.get('why','')}")
    ok = input("按这个计划执行吗？(回车执行 / 输入n取消)：")
    if ok.strip().lower() == "n":
        print("已取消。")
    else:
        results = execute_plan(plan)
        print("\n[综合回答中...]\n")
        print(synthesize(q, plan, results))
input("\n任务完成，按回车退出...")
