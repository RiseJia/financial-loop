"""DataService：数据获取的唯一编排入口。

流程（日线）：
  1. 查本地 parquet 缓存（TTL 内直接返回，含质量检查结果）
  2. 主源 yfinance（带指数退避重试）
  3. 主源失败或质量校验拒绝 → 按市场路由的备源降级
     （.TW/.TWO 台湾官方、.KS Naver、美股 Tiingo→Stooq，见 providers/regional.py）
  4. validate_ohlcv 质量校验：error 级别拒绝该数据，warning 级别放行并记录
  5. 全部源被拒 → 最后手段 repair_ohlcv 孤立坏行修复（≤ max(5行, 0.5%) 才修，
     修复行为记入 warnings 显形）——干净的备源数据永远优先于修补过的主源数据
  6. 写缓存

cross_check(ticker) 提供双源对账，供 `finloop quality` 命令与定期巡检使用。
"""

from __future__ import annotations

import logging

import pandas as pd

from . import cache
from .providers import Provider, RegionalFallbackProvider, YFinanceProvider
from .providers.base import empty_ohlcv
from .quality import QualityReport, reconcile, repair_ohlcv, validate_ohlcv

log = logging.getLogger("finloop.data")


class DataService:
    def __init__(self, primary: Provider | None = None,
                 fallback: Provider | None = None, use_cache: bool = True):
        self.primary = primary or YFinanceProvider()
        self.fallback = fallback or RegionalFallbackProvider()
        self.use_cache = use_cache
        self.last_report: QualityReport | None = None
        self.last_source: str | None = None
        self.last_adjustment: str | None = None  # full/split_only/none——回测收益率口径

    # ------------------------------------------------------------ 日线
    def get_daily(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        if self.use_cache:
            cached = cache.load(ticker, "1d", period)
            if cached is not None:
                report = validate_ohlcv(cached, ticker)
                if report.ok:  # 缓存同样过质量门：被篡改/损坏的缓存视为未命中
                    self.last_source, self.last_report = "cache", report
                    self.last_adjustment = None  # 缓存未记口径（cache-convention-blind）
                    return cached

        rejected: list[tuple[str, pd.DataFrame]] = []
        for provider in (self.primary, self.fallback):
            try:
                df = provider.fetch_daily(ticker, period)
            except Exception as exc:
                log.warning("%s fetch_daily(%s) 失败: %s", provider.name, ticker, exc)
                continue
            report = validate_ohlcv(df, ticker)
            if report.ok:
                self.last_source, self.last_report = provider.name, report
                # 复权口径：regional 备源随路由变化，读其 last_adjustment
                self.last_adjustment = (getattr(provider, "last_adjustment", None)
                                        or getattr(provider, "adjustment", None))
                if report.warnings:
                    log.warning(report.summary())
                if self.use_cache:
                    cache.save(df, ticker, "1d", period)
                return df
            log.warning("%s 数据被质量校验拒绝: %s", provider.name, report.summary())
            if not df.empty:
                rejected.append((provider.name, df))

        # 全部源被拒后的最后手段：孤立坏行修复（干净的备源永远优先于修补过的主源）。
        # repair_ohlcv 自带"坏行极少才修"的纪律，修复行为记入 warnings 显形。
        for name, df in rejected:
            repaired, notes = repair_ohlcv(df, ticker)
            if not notes:
                continue
            report = validate_ohlcv(repaired, ticker)
            if report.ok:
                report.warnings = notes + report.warnings
                self.last_source, self.last_report = f"{name}+repair", report
                log.warning(report.summary())
                if self.use_cache:
                    cache.save(repaired, ticker, "1d", period)
                return repaired

        self.last_source, self.last_report = None, QualityReport(ticker, errors=["全部数据源失败"])
        return empty_ohlcv()

    # ------------------------------------------------------------ 分钟线
    def get_intraday(self, ticker: str, interval: str = "5m",
                     period: str = "5d") -> pd.DataFrame:
        if self.use_cache:
            cached = cache.load(ticker, interval, period)
            if cached is not None:
                report = validate_ohlcv(cached, ticker, daily=False)
                if report.ok:
                    self.last_source, self.last_report = "cache", report
                    return cached
        try:
            df = self.primary.fetch_intraday(ticker, interval, period)
        except Exception as exc:
            log.warning("%s fetch_intraday(%s) 失败: %s", self.primary.name, ticker, exc)
            return empty_ohlcv()
        report = validate_ohlcv(df, ticker, daily=False)
        self.last_source, self.last_report = self.primary.name, report
        if not report.ok:
            log.warning(report.summary())
            return empty_ohlcv()
        if self.use_cache:
            cache.save(df, ticker, interval, period)
        return df

    # ------------------------------------------------------------ 对账
    def cross_check(self, ticker: str, period: str = "6mo",
                    tolerance: float = 0.01) -> dict:
        """双源独立拉取并对账（绕过缓存）。

        容差说明：yfinance 全调整 vs stooq 仅拆股调整，近端（最近一次分红前）
        应几乎一致；存在分红的区间会有等于累计股息率的系统性偏差。
        """
        out: dict = {"ticker": ticker, "primary": self.primary.name,
                     "secondary": self.fallback.name}
        try:
            p = self.primary.fetch_daily(ticker, period)
        except Exception as exc:
            out["error"] = f"主源失败: {exc}"
            return out
        try:
            s = self.fallback.fetch_daily(ticker, period)
        except Exception as exc:
            out["error"] = f"备源失败: {exc}"
            s = None
        # 路由名在发请求前就已写入，失败路径也要带上——排错需要知道死的是哪个子源
        route = getattr(self.fallback, "last_route", None)
        if route:
            out["secondary"] = f"{self.fallback.name}:{route}"
        if s is None:
            return out
        out["primary_quality"] = validate_ohlcv(p, ticker)
        out["secondary_quality"] = validate_ohlcv(s, ticker)
        out.update(reconcile(p, s, tolerance=tolerance))
        return out


# 模块级单例：普通使用路径全部走它
default_service = DataService()
