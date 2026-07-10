"""市场状态机（regime classification）。

先判断「现在是什么市场」，再决定「该用哪类策略」——
这是整个框架最重要的元规则：趋势市用趋势跟随，震荡市用均值回归，
转折期降低仓位等待确认。

状态分类（基于日线）：
  trend_up      上升趋势市：ADX≥25 且 +DI>-DI 且价格在 200 日线上
  trend_down    下降趋势市：ADX≥25 且 -DI>+DI 且价格在 200 日线下
  range         震荡市：ADX<20
  turning       转折期：其余情况（趋势强度与方向证据互相矛盾）
"""

from __future__ import annotations

import pandas as pd

STRATEGY_HINTS = {
    "trend_up": (
        "上升趋势市",
        "顺势而为：趋势跟随策略占优。回调至 20/50 日均线或 VWAP 的低吸优于追高；"
        "RSI 超买在此环境下会钝化，不宜作为做空依据；移动止盈（如跌破 20 日线）优于固定目标价。"
    ),
    "trend_down": (
        "下降趋势市",
        "防御优先：下跌趋势中的超卖反弹大多是出货机会而非买点。长线资金等待趋势修复"
        "（收复 200 日线 + 均线结构改善）再分批介入；短线做多需快进快出。"
    ),
    "range": (
        "震荡市",
        "均值回归策略占优：区间下沿/布林下轨/RSI<30 低吸，区间上沿/布林上轨/RSI>70 止盈。"
        "趋势类信号（金叉死叉）在震荡市中可靠度低，注意布林挤压——震荡的尽头往往是趋势的开始。"
    ),
    "turning": (
        "转折期",
        "证据相互矛盾，是胜率最低的环境：降低仓位、缩小单笔风险，等待状态明朗。"
        "重点观察：均线结构能否修复/恶化、动量切换是否完成、放量方向。"
    ),
}


def _raw_regime(row) -> str:
    """单根 K 线的原始状态（无滞回）。"""
    adx_, plus, minus = row["adx"], row["plus_di"], row["minus_di"]
    above200 = bool(row["close"] > row["sma200"]) if pd.notna(row["sma200"]) else None
    if adx_ >= 25 and plus > minus and above200:
        return "trend_up"
    if adx_ >= 25 and minus > plus and above200 is False:
        return "trend_down"
    if adx_ < 20:
        return "range"
    return "turning"


def classify_regime(df: pd.DataFrame, persist: int = 3) -> dict:
    """返回 {regime, label, hint, evidence}，带滞回确认。

    滞回：原始状态必须连续 persist 根一致才被采纳，否则维持上一个已确认状态。
    这消除价格在 200 日线附近、或 ADX/DI 交叉处的逐日抖动——否则下游策略选择
    （趋势跟随 vs 均值回归）会自相矛盾地日日翻转（审计 regime-hysteresis 修复）。
    """
    row = df.iloc[-1]
    adx_, plus, minus = row["adx"], row["plus_di"], row["minus_di"]
    above200 = bool(row["close"] > row["sma200"]) if pd.notna(row["sma200"]) else None

    # 对最近一段逐根求原始状态，做持续确认
    tail = df.iloc[-(persist * 4):] if len(df) >= persist else df
    raw = [_raw_regime(tail.iloc[i]) for i in range(len(tail))]
    confirmed = raw[0]
    run_val, run_len = raw[0], 1
    for r in raw[1:]:
        if r == run_val:
            run_len += 1
        else:
            run_val, run_len = r, 1
        if run_len >= persist:
            confirmed = run_val
    regime = confirmed

    evidence = [
        f"ADX = {adx_:.1f}（{'≥25 有趋势' if adx_ >= 25 else '<20 无趋势' if adx_ < 20 else '20-25 灰色地带'}）",
        f"+DI {plus:.1f} vs -DI {minus:.1f}（{'上行' if plus > minus else '下行'}方向占优）",
    ]
    if above200 is not None:
        evidence.append(f"价格{'高于' if above200 else '低于'} 200 日均线（长线{'牛市' if above200 else '熊市'}结构）")
    if regime != _raw_regime(row):
        evidence.append(f"（滞回：当根原始状态为 {_raw_regime(row)}，未满 {persist} 根确认，维持 {regime}）")

    label, hint = STRATEGY_HINTS[regime]
    return {"regime": regime, "label": label, "hint": hint, "evidence": evidence}
