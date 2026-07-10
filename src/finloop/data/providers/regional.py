"""按市场路由的组合备源（DataService 的默认 fallback）。

为什么需要它：旧默认备源 Stooq 有两个致命问题——
  1. 只支持美股（带后缀的日台韩欧标的从来就没有过备源）；
  2. 2026-06 起对数据中心 IP 部署了 JS 反爬验证，云端环境不可用。

路由表（按 ticker 后缀）：
  .TW / .TWO   → TaiwanProvider（证交所/柜买中心官方开放数据，无 key）
  .KS          → NaverProvider（韩国事实标准行情门户，无 key）
  无后缀美股   → TiingoProvider（免费注册 key，环境变量 TIINGO_API_KEY）；
                 无 key 时退回 StooqProvider 作最后手段（可能被反爬墙拦）
  其他市场（.T/.HK/.DE/.VI/.SW/.SZ/.AS/.KQ…）→ 暂无备源，诚实返回空
                 （日本可后续接 J-Quants，需免费注册，本轮未实现）

设计约束：路由只看后缀、不发请求；具体 provider 的网络失败由
DataService 统一捕获记录。last_route 记录最近一次实际使用的子源名，
供对账输出与排错。
"""

from __future__ import annotations

import pandas as pd

from .base import empty_ohlcv
from .naver_provider import NaverProvider
from .stooq_provider import StooqProvider
from .taiwan_provider import TaiwanProvider
from .tiingo_provider import TiingoProvider


class RegionalFallbackProvider:
    name = "regional-fallback"
    adjustment = "mixed"  # 实际口径随路由子源而变，见 last_adjustment

    def __init__(self, tiingo: TiingoProvider | None = None,
                 taiwan: TaiwanProvider | None = None,
                 naver: NaverProvider | None = None,
                 stooq: StooqProvider | None = None):
        self.tiingo = tiingo if tiingo is not None else TiingoProvider()
        self.taiwan = taiwan if taiwan is not None else TaiwanProvider()
        self.naver = naver if naver is not None else NaverProvider()
        self.stooq = stooq if stooq is not None else StooqProvider()
        self.last_route: str | None = None
        self.last_adjustment: str | None = None

    def _route(self, ticker: str):
        if ticker.endswith((".TW", ".TWO")):
            return self.taiwan
        if ticker.endswith(".KS"):
            return self.naver
        if not any(ch in ticker for ch in "^=."):
            return self.tiingo if self.tiingo.available else self.stooq
        return None

    def fetch_daily(self, ticker: str, period: str) -> pd.DataFrame:
        provider = self._route(ticker)
        self.last_route = provider.name if provider else None
        self.last_adjustment = getattr(provider, "adjustment", None) if provider else None
        if provider is None:
            return empty_ohlcv()
        return provider.fetch_daily(ticker, period)

    def fetch_intraday(self, ticker: str, interval: str, period: str) -> pd.DataFrame:
        return empty_ohlcv()  # 备源只做日线
