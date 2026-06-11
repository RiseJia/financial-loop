"""情景压力测试套件：用可控的合成行情检验策略的状态依赖性。

量化研究中，单一历史路径的回测只是一次抽样；情景套件回答的是更结构化的问题：
「这个策略在哪类市场状态下赚钱、在哪类状态下亏钱、亏多少」。

每个情景生成 N 条独立随机路径（不同 seed），统计分布而非单点。
参数大致校准到美股大盘的量级（年化漂移/波动）。

情景设计的预期答案（用于交叉检验引擎与策略实现是否自洽）：
  secular_bull    长牛：趋势策略应接近买入持有，均值回归吃亏
  grinding_bear   阴跌熊市：趋势过滤类策略的主战场（少亏即是赢）
  crash_recovery  崩盘-修复：检验回撤控制与重新进场的及时性
  range_bound     箱体震荡：均值回归占优，趋势策略被反复止损
  whipsaw         高波动来回打脸：所有趋势策略的最坏情景，检验亏损是否有界
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _gbm(n: int, annual_drift: float, annual_vol: float, rng: np.random.Generator,
         s0: float = 100.0) -> np.ndarray:
    """几何布朗运动日收益路径。"""
    mu, sigma = annual_drift / TRADING_DAYS, annual_vol / np.sqrt(TRADING_DAYS)
    rets = rng.normal(mu, sigma, n)
    return s0 * np.cumprod(1 + rets)


def secular_bull(rng: np.random.Generator) -> np.ndarray:
    """5 年长牛：年化 +14%，波动 16%。"""
    return _gbm(5 * TRADING_DAYS, 0.14, 0.16, rng)


def grinding_bear(rng: np.random.Generator) -> np.ndarray:
    """2 年阴跌（年化 -22%，波动 26%）+ 1 年弱修复（+8%）。"""
    p1 = _gbm(2 * TRADING_DAYS, -0.22, 0.26, rng)
    p2 = _gbm(1 * TRADING_DAYS, 0.08, 0.20, rng, s0=p1[-1])
    return np.concatenate([p1, p2])


def crash_recovery(rng: np.random.Generator) -> np.ndarray:
    """1.5 年正常牛 → 2 个月 -35% 级崩盘（波动 55%）→ 1.5 年强修复。"""
    p1 = _gbm(int(1.5 * TRADING_DAYS), 0.12, 0.15, rng)
    # 漂移 -2.2/年 × 42天 ≈ 累计 -31%（含波动拖累约 -33%），对标 2020-03 量级
    crash = _gbm(42, -2.2, 0.55, rng, s0=p1[-1])
    p3 = _gbm(int(1.5 * TRADING_DAYS), 0.25, 0.22, rng, s0=crash[-1])
    return np.concatenate([p1, crash, p3])


def range_bound(rng: np.random.Generator) -> np.ndarray:
    """3 年箱体：OU 均值回归过程，围绕 100 震荡（半衰期约 1 个月）。"""
    n = 3 * TRADING_DAYS
    theta, sigma = 0.03, 0.013
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] - theta * x[i - 1] + rng.normal(0, sigma)
    return 100 * np.exp(x)


def whipsaw(rng: np.random.Generator) -> np.ndarray:
    """3 年高波动来回：每 ~6 周翻转一次方向（年化 ±45%），趋势策略最坏情景。"""
    segs = []
    s0, sign = 100.0, 1
    for _ in range(26):
        seg = _gbm(30, sign * 0.45, 0.30, rng, s0=s0)
        segs.append(seg)
        s0, sign = seg[-1], -sign
    return np.concatenate(segs)


SCENARIOS = {
    "secular_bull": (secular_bull, "长牛（年化+14%）"),
    "grinding_bear": (grinding_bear, "阴跌熊市+弱修复"),
    "crash_recovery": (crash_recovery, "崩盘-修复"),
    "range_bound": (range_bound, "箱体震荡"),
    "whipsaw": (whipsaw, "高波动来回打脸"),
}


def make_ohlcv_from_closes(closes: np.ndarray, start="2021-01-04") -> pd.DataFrame:
    """合成路径 → 标准 OHLCV（OHLC 自洽，成交量与 |收益| 弱相关更接近真实）。"""
    idx = pd.bdate_range(start, periods=len(closes))
    close = pd.Series(closes, index=idx, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    body_hi = pd.concat([open_, close], axis=1).max(axis=1)
    body_lo = pd.concat([open_, close], axis=1).min(axis=1)
    rets = close.pct_change().fillna(0).abs()
    volume = 1e6 * (1 + 5 * rets)  # 放量与波动正相关
    return pd.DataFrame({
        "open": open_, "high": body_hi * 1.004, "low": body_lo * 0.996,
        "close": close, "volume": volume,
    })


def generate(scenario: str, n_paths: int = 5, base_seed: int = 2024) -> list[pd.DataFrame]:
    """生成某情景的 n 条独立路径（可复现）。"""
    fn = SCENARIOS[scenario][0]
    return [make_ohlcv_from_closes(fn(np.random.default_rng(base_seed + i)))
            for i in range(n_paths)]
