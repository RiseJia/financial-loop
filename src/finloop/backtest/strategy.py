"""策略抽象与内置示例策略。

================================================================
策略是什么的抽象？
================================================================
一个策略 = 一个映射：

    f: 截至 t 时点的全部已知信息  →  t 时点收盘后的目标仓位

在本框架中具体化为：

    Strategy = Callable[[enrich后的DataFrame], pd.Series]

输入：带全部指标列的 OHLCV（行 = 时间）
输出：目标仓位序列 position ∈ [0, 1]（0=空仓，1=满仓多头；
      做空策略可扩展到 [-1, 1]），索引与输入对齐。

关键契约（因果性）：position[t] 只允许依赖 df 的第 0..t 行。
由于指标全部用 rolling/ewm 因果计算，逐行映射天然满足；
引擎侧还会再 shift(1) 强制「t 的决策最早在 t+1 兑现」，双保险。

把策略约束成这个纯函数形态的好处：
  - 可回测：同一函数喂历史数据 = 历史模拟，喂今天数据 = 实时决策，零分叉；
  - 可比较：所有策略输出同一种东西（仓位序列），指标体系通用；
  - 可组合：仓位序列可以加权平均（组合多策略）、可以叠乘风控开关。
================================================================
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

Strategy = Callable[[pd.DataFrame], pd.Series]


# ---------------------------------------------------------------- 内置策略

def strat_buy_hold(df: pd.DataFrame) -> pd.Series:
    """基准：始终满仓。任何主动策略都必须先证明自己赢过它。"""
    return pd.Series(1.0, index=df.index)


def strat_sma200(df: pd.DataFrame) -> pd.Series:
    """200日线趋势过滤：收盘在 200 日线上方持有，下方空仓。

    长线文档里的核心规则，最经典的单参数择时。
    预期行为：牛市略跑输买入持有（信号滞后），熊市大幅减少回撤。
    """
    return (df["close"] > df["sma200"]).astype(float)


def strat_macd(df: pd.DataFrame) -> pd.Series:
    """MACD 金叉持有、死叉空仓。教学用途：演示一个『听起来合理』的
    信号在扣除成本后经常跑不过基准——信号频率与成本的矛盾。"""
    return (df["macd"] > df["macd_signal"]).astype(float)


def strat_rsi_mr(df: pd.DataFrame) -> pd.Series:
    """RSI 均值回归：RSI<30 买入，RSI>55 卖出（带状态记忆）。

    这是有状态策略的示例：持仓状态取决于历史路径，不能用单行条件
    向量化，需要顺序扫描。预期只在震荡环境有效。
    """
    rsi = df["rsi14"].to_numpy()
    pos = np.zeros(len(df))
    holding = 0.0
    for i in range(len(df)):
        if holding == 0.0 and rsi[i] < 30:
            holding = 1.0
        elif holding == 1.0 and rsi[i] > 55:
            holding = 0.0
        pos[i] = holding
    return pd.Series(pos, index=df.index)


def strat_momentum(df: pd.DataFrame) -> pd.Series:
    """动量切换策略：动量得分 ≥+2 进场，≤-2 离场，过渡区维持原仓位。

    直接复用 signals.momentum_switch 的四分量得分，
    带滞回（hysteresis）——这正是『过渡区不动作』设计的回测版。
    """
    from ..signals.momentum_switch import momentum_score

    score = momentum_score(df).to_numpy()
    pos = np.zeros(len(df))
    holding = 0.0
    for i in range(len(df)):
        if score[i] >= 2:
            holding = 1.0
        elif score[i] <= -2:
            holding = 0.0
        pos[i] = holding
    return pd.Series(pos, index=df.index)


STRATEGIES: dict[str, Strategy] = {
    "buy_hold": strat_buy_hold,
    "sma200": strat_sma200,
    "macd": strat_macd,
    "rsi_mr": strat_rsi_mr,
    "momentum": strat_momentum,
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"未知策略 '{name}'，可用：{', '.join(STRATEGIES)}")
    return STRATEGIES[name]
