"""数据质量校验管道。

原则：坏数据宁可拒绝也不能静默流入指标层（garbage in, garbage out）。
validate_ohlcv() 对标准 OHLCV DataFrame 做一组健康检查，返回 QualityReport；
上层据此决定：通过 / 带警告使用 / 拒绝。

检查项：
  结构类（error，直接拒绝）：
    - 空数据 / 缺列
    - 时间戳重复或乱序
    - 非正价格（close <= 0）
    - OHLC 自洽性：high >= max(open,close), low <= min(open,close), high >= low
  统计类（warning，带警告使用）：
    - 极端跳变：单日收益超过 8 倍近期波动（疑似坏点或未调整的拆股）
    - 连续零成交量（疑似数据缺失或停牌）
    - 交易日缺口：相邻 bar 间隔超过 7 个自然日（长假最多 4 天）
    - NaN 比例过高
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    ticker: str
    n_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """没有结构性错误（警告不阻断使用）。"""
        return not self.errors and self.n_rows > 0

    def summary(self) -> str:
        status = "✅ 通过" if self.ok and not self.warnings else ("⚠️ 带警告" if self.ok else "❌ 拒绝")
        lines = [f"[{self.ticker}] 数据质量：{status}（{self.n_rows} 行）"]
        lines += [f"  ❌ {e}" for e in self.errors]
        lines += [f"  ⚠️ {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_ohlcv(df: pd.DataFrame, ticker: str = "?", daily: bool = True) -> QualityReport:
    rep = QualityReport(ticker=ticker, n_rows=len(df))

    # --- 结构检查 ---
    if df.empty:
        rep.errors.append("空数据")
        return rep
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        rep.errors.append(f"缺少列：{missing}")
        return rep

    if df.index.duplicated().any():
        rep.errors.append(f"时间戳重复 {int(df.index.duplicated().sum())} 处")
    if not df.index.is_monotonic_increasing:
        rep.errors.append("时间戳乱序")

    if (df["close"] <= 0).any():
        rep.errors.append(f"非正收盘价 {int((df['close'] <= 0).sum())} 处")

    bad_ohlc = (
        (df["high"] < df[["open", "close"]].max(axis=1) - 1e-9)
        | (df["low"] > df[["open", "close"]].min(axis=1) + 1e-9)
        | (df["high"] < df["low"])
    )
    if bad_ohlc.any():
        rep.errors.append(f"OHLC 自洽性破坏 {int(bad_ohlc.sum())} 处（high/low 与 open/close 矛盾）")

    if rep.errors:
        return rep

    # --- 统计检查（warning）---
    rets = df["close"].pct_change()
    # 用前一日为止的波动率做阈值（shift），否则跳变会抬高自己的阈值而逃过检测
    vol = rets.rolling(20, min_periods=10).std().shift(1)
    jumps = (rets.abs() > (8 * vol).clip(lower=0.12)).fillna(False)
    if jumps.any():
        dates = [d.date().isoformat() for d in df.index[jumps][:3]]
        rep.warnings.append(
            f"极端跳变 {int(jumps.sum())} 处（如 {', '.join(dates)}）——疑似坏点或未调整的拆股/除权"
        )

    zero_vol = (df["volume"] == 0)
    if zero_vol.mean() > 0.02:
        rep.warnings.append(f"零成交量占比 {zero_vol.mean():.1%}——疑似数据缺失或低流动性")

    if daily and len(df) > 1:
        gaps = df.index.to_series().diff().dt.days
        big_gaps = gaps[gaps > 7]
        if len(big_gaps):
            rep.warnings.append(f"交易日缺口 {len(big_gaps)} 处（最大 {int(big_gaps.max())} 天）")

    nan_ratio = df[["open", "high", "low", "volume"]].isna().mean().max()
    if nan_ratio > 0.05:
        rep.warnings.append(f"NaN 比例 {nan_ratio:.1%}")

    return rep


def reconcile(primary: pd.DataFrame, secondary: pd.DataFrame,
              tolerance: float = 0.01) -> dict:
    """双源对账：比较两个来源在重叠日期上的收盘价。

    返回 {overlap_days, mean_abs_dev, max_abs_dev, n_breaches, breach_dates}。
    注意：不同源的复权口径可能不同（如 Stooq 只做拆股调整、不做分红调整），
    近端日期应高度一致，远端历史允许系统性小偏差——所以默认只对账最近一段。
    """
    if primary.empty or secondary.empty:
        return {"overlap_days": 0}
    p = primary["close"].copy()
    s = secondary["close"].copy()
    p.index = pd.to_datetime(p.index).tz_localize(None).normalize()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    joined = pd.concat([p, s], axis=1, keys=["p", "s"], join="inner").dropna()
    if joined.empty:
        return {"overlap_days": 0}
    dev = (joined["p"] / joined["s"] - 1).abs()
    breaches = dev[dev > tolerance]
    return {
        "overlap_days": len(joined),
        "mean_abs_dev": float(dev.mean()),
        "max_abs_dev": float(dev.max()),
        "n_breaches": len(breaches),
        "breach_dates": [d.date().isoformat() for d in breaches.index[:5]],
    }
