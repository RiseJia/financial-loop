"""新闻抓取：通过 yfinance 获取个股 / 大盘相关头条。"""

from __future__ import annotations

import yfinance as yf


def get_news(ticker: str, limit: int = 5) -> list[dict]:
    """返回 [{title, publisher, link, time}]，失败时返回空列表。"""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    items = []
    for n in raw[:limit]:
        # yfinance 新旧版本的新闻结构不同，做兼容处理
        content = n.get("content", n)
        title = content.get("title")
        if not title:
            continue
        provider = content.get("provider") or {}
        link = (content.get("canonicalUrl") or {}).get("url") or n.get("link", "")
        items.append({
            "title": title,
            "publisher": provider.get("displayName") or n.get("publisher", ""),
            "link": link,
            "time": content.get("pubDate") or "",
        })
    return items
