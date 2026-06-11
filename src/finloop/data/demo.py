"""数据质量管线演示：`finloop quality --demo`。

离线、确定性地展示管线对每种已知故障模式的处置：
对干净的合成行情逐一注入真实世界会发生的数据故障，
展示校验器的判定（拒绝/警告/放行）与 DataService 的降级行为。

故障模式与真实来源对照：
  非正价格        来源：API 返回错误记录、停牌占位
  high < low      来源：拼接/复权错误（yfinance 历史上发生过）
  时间戳重复      来源：分页拉取重叠、时区折叠
  极端跳变        来源：未调整的拆股（例如 10:1 拆股日 -90%）、坏 tick
  零成交量段      来源：数据缺失、退市边缘标的
  交易日缺口      来源：API 部分返回
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .providers.base import empty_ohlcv
from .quality import validate_ohlcv
from .service import DataService


def _clean(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series(closes, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    hi = pd.concat([open_, close], axis=1).max(axis=1) * 1.004
    lo = pd.concat([open_, close], axis=1).min(axis=1) * 0.996
    vol = pd.Series(1e6, index=idx) * (1 + 5 * close.pct_change().abs().fillna(0))
    return pd.DataFrame({"open": open_, "high": hi, "low": lo,
                         "close": close, "volume": vol})


def _corruptions(base: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    cases = []

    df = base.copy()
    df.iloc[50, df.columns.get_loc("close")] = -3.0
    cases.append(("非正价格（API 坏记录）", df))

    df = base.copy()
    df.iloc[80, df.columns.get_loc("high")] = df.iloc[80]["low"] - 5
    cases.append(("high < low（复权/拼接错误）", df))

    df = pd.concat([base, base.iloc[[120]]]).sort_index()
    cases.append(("时间戳重复（分页重叠）", df))

    df = base.copy()
    jump_loc = df.columns.get_loc("close")
    df.iloc[150:, jump_loc] = df.iloc[150:, jump_loc].to_numpy() * 0.1
    hi_loc, lo_loc, op_loc = (df.columns.get_loc(c) for c in ("high", "low", "open"))
    df.iloc[150:, hi_loc] = df.iloc[150:, hi_loc].to_numpy() * 0.1
    df.iloc[150:, lo_loc] = df.iloc[150:, lo_loc].to_numpy() * 0.1
    df.iloc[151:, op_loc] = df.iloc[151:, op_loc].to_numpy() * 0.1
    df.iloc[150, op_loc] = df.iloc[150]["close"] * 1.02  # 拆股当日 open 仍接近新价
    df.iloc[150, hi_loc] = max(df.iloc[150]["open"], df.iloc[150]["close"]) * 1.004
    df.iloc[150, lo_loc] = min(df.iloc[150]["open"], df.iloc[150]["close"]) * 0.996
    cases.append(("未调整的 10:1 拆股（-90% 跳变）", df))

    df = base.copy()
    df.iloc[200:230, df.columns.get_loc("volume")] = 0.0
    cases.append(("连续零成交量段", df))

    df = pd.concat([base.iloc[:100], base.iloc[130:]])
    cases.append(("交易日缺口（API 部分返回）", df))

    return cases


class _StaticProvider:
    def __init__(self, name: str, df: pd.DataFrame):
        self.name, self.df = name, df

    def fetch_daily(self, ticker, period):
        return self.df

    def fetch_intraday(self, ticker, interval, period):
        return empty_ohlcv()


def run_demo() -> str:
    base = _clean()
    lines = ["=" * 64,
             "数据质量管线演示（离线确定性，故障注入）",
             "=" * 64, "",
             "① 基线：干净数据"]
    lines.append("   " + validate_ohlcv(base, "CLEAN").summary().replace("\n", "\n   "))

    lines += ["", "② 故障注入 → 校验器判定："]
    for label, corrupted in _corruptions(base):
        rep = validate_ohlcv(corrupted, "DEMO")
        verdict = "🚫 拒绝(触发降级)" if not rep.ok else ("⚠️ 警告放行" if rep.warnings else "✅ 通过")
        lines.append(f"\n   ▸ 注入：{label}")
        lines.append(f"     判定：{verdict}")
        for e in rep.errors:
            lines.append(f"       ❌ {e}")
        for w in rep.warnings:
            lines.append(f"       ⚠️ {w}")

    lines += ["", "③ DataService 降级演示：主源被污染（high<low）→ 自动切换备源"]
    bad = base.copy()
    bad.iloc[80, bad.columns.get_loc("high")] = bad.iloc[80]["low"] - 5
    svc = DataService(_StaticProvider("corrupted-primary", bad),
                      _StaticProvider("clean-fallback", base), use_cache=False)
    df = svc.get_daily("DEMO")
    lines.append(f"   结果：获得 {len(df)} 行干净数据，实际来源 = {svc.last_source}")

    lines += ["", "④ 全源失败演示：返回空数据 + 拒绝报告（绝不静默给坏数据）"]
    svc2 = DataService(_StaticProvider("p", bad), _StaticProvider("f", bad), use_cache=False)
    df2 = svc2.get_daily("DEMO")
    lines.append(f"   结果：{len(df2)} 行，报告：{'; '.join(svc2.last_report.errors)}")

    lines += ["", "=" * 64,
              "在有网络的环境对真实数据执行：finloop quality NVDA",
              "=" * 64]
    return "\n".join(lines)
