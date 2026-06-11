"""主数据源：yfinance（Yahoo Finance 非官方接口）。

特性：免费、日线+分钟线+基本面+新闻全覆盖。
缺陷：无 SLA、可能限流、auto_adjust 全调整价格但不调整成交量。
"""

from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from .base import empty_ohlcv, normalize_ohlcv


class YFinanceProvider:
    name = "yfinance"

    def __init__(self, retries: int = 3, backoff: float = 2.0):
        self.retries = retries
        self.backoff = backoff

    def _fetch(self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                df = yf.Ticker(ticker).history(
                    period=period, interval=interval, auto_adjust=True
                )
                return normalize_ohlcv(df)
            except Exception as exc:
                last_exc = exc
                time.sleep(self.backoff * (2 ** attempt))
        if last_exc:
            raise last_exc
        return empty_ohlcv()

    def fetch_daily(self, ticker: str, period: str) -> pd.DataFrame:
        return self._fetch(ticker, "1d", period)

    def fetch_intraday(self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        return self._fetch(ticker, interval, period)
