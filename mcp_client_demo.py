"""任务9：MCP Client —— 连上 Server，像调本地函数一样调远程工具

注意体会：Client 侧完全不知道 get_weather 背后是哪个天气 API、
requests 还是别的实现 —— 它只认 MCP 协议。
"""
import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # 注：mcp 2.x 的 stdio 子进程只继承白名单环境变量，
    # 如果 Server 需要 key/代理，得用 env={...} 显式传；本例 Server 免 key，不用传
    params = StdioServerParameters(
        command=sys.executable,  # 用当前这个 python，比写死 "python" 更稳
        args=["mcp_weather_server.py"],  # 和 Client 同一个目录
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Server 提供的工具：", [t.name for t in tools.tools])

            r = await session.call_tool("get_weather", {"city": "北京"})
            print("调 get_weather 结果：", r.content[0].text)

            r2 = await session.call_tool("get_today", {})
            print("调 get_today 结果：", r2.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
