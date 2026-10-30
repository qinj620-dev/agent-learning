"""任务9：MCP Server —— 把工具包装成标准服务

原来：get_weather 是 weather_agent.py 里的一个本地函数，
     换个 Agent 想用，得把代码拷过去再手写一遍对接。
现在：把它注册成 MCP 工具，任何 MCP Client 连上就能直接用，
     不用改一行对接代码 —— 工具界的 USB 接口。

数据源用 Open-Meteo：免费、免 key，这课专注学 MCP 本身，
不用为注册 key 分心。
"""
import requests
# mcp 2.x 里 FastMCP 改名叫 MCPServer，老教程里看到 FastMCP 是同一个东西
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("天气服务")

_WEATHER_CN = {0: "晴", 1: "大致晴", 2: "多云", 3: "阴",
               45: "雾", 48: "雾凇",
               51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨",
               61: "小雨", 63: "中雨", 65: "大雨",
               71: "小雪", 73: "中雪", 75: "大雪",
               80: "阵雨", 81: "阵雨", 82: "暴雨",
               95: "雷暴", 96: "雷暴伴冰雹", 99: "雷暴伴冰雹"}


@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的实时天气"""
    g = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "zh", "format": "json"},
        timeout=15).json()
    if not g.get("results"):
        return f"找不到城市：{city}"
    r0 = g["results"][0]
    w = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": r0["latitude"], "longitude": r0["longitude"],
                "current": "temperature_2m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min",
                "timezone": "auto"},
        timeout=15).json()
    cur, daily = w["current"], w["daily"]
    desc = _WEATHER_CN.get(cur["weather_code"], "未知")
    return (f"{r0.get('name', city)}当前 {cur['temperature_2m']}°C，{desc}，"
            f"今日 {daily['temperature_2m_min'][0]}~{daily['temperature_2m_max'][0]}°C")


@mcp.tool()
def get_today() -> str:
    """获取今天的日期和星期"""
    from datetime import date
    return date.today().strftime("%Y-%m-%d %A")


if __name__ == "__main__":
    mcp.run()  # 默认走 stdio：Client 以子进程方式把它拉起来连上
