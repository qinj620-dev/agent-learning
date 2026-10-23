"""任务9：MCP Server —— 把工具包装成标准服务

原来：get_weather 是 weather_agent.py 里的一个本地函数，
     换个 Agent 想用，得把代码拷过去再手写一遍对接。
现在：把它注册成 MCP 工具，任何 MCP Client 连上就能直接用，
     不用改一行对接代码 —— 工具界的 USB 接口。
"""
import os
import requests
# mcp 2.x 里 FastMCP 改名叫 MCPServer，老教程里看到 FastMCP 是同一个东西
from mcp.server.mcpserver import MCPServer

QWEATHER_KEY = os.getenv("QWEATHER_KEY")

mcp = MCPServer("天气服务")


@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的实时天气"""
    loc = requests.get(
        "https://geoapi.qweather.com/v2/city/lookup",
        params={"location": city, "key": QWEATHER_KEY}, timeout=15).json()
    if not loc.get("location"):
        return f"找不到城市：{city}"
    lid = loc["location"][0]["id"]
    w = requests.get(
        "https://devapi.qweather.com/v7/weather/now",
        params={"location": lid, "key": QWEATHER_KEY}, timeout=15).json()["now"]
    return f"{city}当前 {w['temp']}°C，{w['text']}，体感 {w['feelsLike']}°C"


@mcp.tool()
def get_today() -> str:
    """获取今天的日期和星期"""
    from datetime import date
    return date.today().strftime("%Y-%m-%d %A")


if __name__ == "__main__":
    mcp.run()  # 默认走 stdio：Client 以子进程方式把它拉起来连上
