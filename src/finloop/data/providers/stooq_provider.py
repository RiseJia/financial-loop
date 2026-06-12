"""备用数据源：Stooq（https://stooq.com，免费 CSV 接口，无需 API key）。

⚠️ 状态（2026-06 实测）：stooq.com 已对数据中心 IP 部署 JS 工作量证明
反爬验证，程序化访问拿到的是验证页而非 CSV——本 provider 在云端环境
基本不可用，已从默认备源降级为美股的最后手段（见 regional.py 的路由链）。
住宅网络/本地环境可能仍然可用，故保留。

用途（可用时）：
  1. yfinance 失败时的美股日线降级源；
  2. 双源对账（reconcile）的独立参照。
限制：只有日线；只做拆股调整、不做分红调整（与 yfinance 的 auto_adjust
全调整口径存在系统性小偏差，对账容差需考虑这一点）；美股代码需加 .us 后缀。
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from .base import PERIOD_DAYS, empty_ohlcv, normalize_ohlcv


def _stooq_symbol(ticker: str) -> str | None:
    """美股普通代码 → stooq 代码；指数/期货等特殊代码不支持，返回 None。"""
    if any(ch in ticker for ch in "^=-."):
        return None
    return f"{ticker.lower()}.us"


class StooqProvider:
    name = "stooq"

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def fetch_daily(self, ticker: str, period: str) -> pd.DataFrame:
        symbol = _stooq_symbol(ticker)
        if symbol is None:
            return empty_ohlcv()
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        text = resp.text
        if not text.startswith("Date,"):
            return empty_ohlcv()  # 未知代码时 stooq 返回提示文本而非 CSV
        df = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
        df = normalize_ohlcv(df)
        days = PERIOD_DAYS.get(period, 731)
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
        return df[df.index >= cutoff]

    def fetch_intraday(self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        return empty_ohlcv()  # stooq 无分钟线
