"""备用数据源：Tiingo（https://www.tiingo.com，美股日线，免费注册 API key）。

为什么选它替代 Stooq 做美股备源：
  - 官方 API，无反爬墙（Stooq 已对数据中心 IP 部署 JS 验证，见 stooq_provider）；
  - 免费档对学习用途足够（约 50 符号/小时、1000 请求/天，以官网为准）；
  - 提供 adj* 全复权字段（拆股+分红），与 yfinance auto_adjust 同口径——
    双源对账不再有 Stooq 时代"分红区间系统性偏差"的噪音。

启用方式：注册后把 key 放进环境变量 TIINGO_API_KEY。未设置 key 时本
provider 自动报告不可用（available=False），不发任何网络请求。
限制：只覆盖美股/部分 ETF；只有日线（免费档分钟线另有限制，未接入）。

实网验证（2026-06-12）：通过，但首跑即抓到一个盲写 bug——真实响应里
原始列与 adj* 列并存，直接 rename 产生重复列名（修复见 fetch_daily 内
注释与回归测试）。修复后 MU 对账：124 天重叠，平均偏差 0.003%，
最大 0.005%（同为全复权口径，仅剩复权因子舍入噪音）。
"""

from __future__ import annotations

import os

import pandas as pd
import requests

from .base import empty_ohlcv, normalize_ohlcv, period_to_days


def _tiingo_symbol(ticker: str) -> str | None:
    """美股普通代码原样使用；带交易所后缀/指数等特殊代码不支持。"""
    if any(ch in ticker for ch in "^=."):
        return None
    return ticker.upper()


class TiingoProvider:
    name = "tiingo"

    def __init__(self, api_key: str | None = None, timeout: float = 15.0):
        self.api_key = api_key if api_key is not None else os.environ.get("TIINGO_API_KEY", "")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def fetch_daily(self, ticker: str, period: str) -> pd.DataFrame:
        symbol = _tiingo_symbol(ticker)
        if symbol is None or not self.available:
            return empty_ohlcv()
        start = (pd.Timestamp.today().normalize()
                 - pd.Timedelta(days=period_to_days(period))).date().isoformat()
        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        resp = requests.get(
            url,
            params={"startDate": start, "format": "json"},
            headers={"Authorization": f"Token {self.api_key}",
                     "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            return empty_ohlcv()
        df = pd.DataFrame(rows)
        # 用 adj* 全复权列对齐 yfinance auto_adjust 口径
        needed = ["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]
        if not set(needed).issubset(df.columns):
            return empty_ohlcv()
        # 真实响应里原始列（open/close/…）与 adj* 列并存：必须先只挑 adj*，
        # 否则 rename 后出现重复列名，下游 df["close"] 取出两列（2026-06 实网验收抓到）
        df = df[needed]
        df = df.rename(columns={"adjOpen": "open", "adjHigh": "high",
                                "adjLow": "low", "adjClose": "close",
                                "adjVolume": "volume"})
        df.index = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return normalize_ohlcv(df)

    def fetch_intraday(self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        return empty_ohlcv()
