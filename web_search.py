"""联网搜索第三档（2026-11-11 加）：知识库查不到时，先联网搜一圈。

用 Tavily Search API（https://api.tavily.com/search），key 走环境变量
TAVILY_API_KEY，由用户自己注册、自己填到 Render 里，助手不经手。
没配 key 或请求失败时返回 []，调用方回落到大模型知识，不报错、不卡住。
免费额度每月 1000 次；只有库外问题才搜，平时基本不消耗。
"""
import os

import requests

TAVILY_API_URL = "https://api.tavily.com/search"
TIMEOUT = 10


def search_web(query, max_results=5):
    """调 Tavily 搜一圈。返回 [{title, url, snippet}]；没 key 或出错返回 []。"""
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        return []
    try:
        r = requests.post(TAVILY_API_URL, json={
            "api_key": key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
        }, timeout=TIMEOUT)
        if r.status_code != 200:
            print(f"  [联网搜索] Tavily 返回 {r.status_code}，降级为大模型知识")
            return []
        out = []
        for item in r.json().get("results", [])[:max_results]:
            out.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            })
        return out
    except Exception as e:
        print(f"  [联网搜索] 请求失败（{e}），降级为大模型知识")
        return []
