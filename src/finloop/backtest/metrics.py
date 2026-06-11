"""回测绩效指标。

读法速查（详细解释见 docs/backtesting.md）：
  CAGR        年化复合收益——「这条曲线相当于每年赚多少」
  vol         年化波动率——收益的不确定性
  sharpe      夏普 = 年化收益/年化波动——每承担 1 单位波动换来多少收益，
              个股策略 >0.8 已不错，>1.5 就要怀疑哪里算错了
  max_drawdown 最大回撤——从峰值到谷底的最大跌幅，决定「拿不拿得住」
  calmar      CAGR/|最大回撤|——收益与最痛时刻之比
  max_underwater_days 最长水下期——净值创新高之间最久隔了多少交易日，
              心理上这比回撤深度更难熬
  win_rate    日度胜率——注意：胜率低的策略未必差（趋势策略常 <50%
              但单次盈利大），必须与盈亏分布一起看
  exposure    持仓暴露——平均仓位，衡量资金利用率与「躲掉了多少风险」
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_metrics(returns: pd.Series, positions: pd.Series | None = None) -> dict:
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        return {k: 0.0 for k in
                ("total_return", "cagr", "vol", "sharpe", "max_drawdown",
                 "calmar", "win_rate", "exposure")} | {"max_underwater_days": 0}

    equity = (1 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else 0.0
    vol = float(returns.std() * np.sqrt(TRADING_DAYS))
    # 标准定义：年化算术均值收益 / 年化波动（rf=0），而非 CAGR/vol
    ann_mean = float(returns.mean() * TRADING_DAYS)
    sharpe = float(ann_mean / vol) if vol > 1e-12 else 0.0

    # 峰值必须计入初始资本 1.0，否则首段亏损被低估（首日 -40% 会被报成回撤 0）
    peak = equity.cummax().clip(lower=1.0)
    drawdown = equity / peak - 1
    max_dd = float(drawdown.min())

    underwater = drawdown < -1e-12
    max_uw, cur = 0, 0
    for flag in underwater:
        cur = cur + 1 if flag else 0
        max_uw = max(max_uw, cur)

    active = returns[returns != 0]
    win_rate = float((active > 0).mean()) if len(active) else 0.0
    exposure = float(positions.abs().mean()) if positions is not None else 1.0
    calmar = float(cagr / abs(max_dd)) if max_dd < -1e-12 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "max_underwater_days": max_uw,
        "win_rate": win_rate,
        "exposure": exposure,
    }
