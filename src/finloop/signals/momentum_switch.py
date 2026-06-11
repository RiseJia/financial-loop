"""动量切换（momentum switching）识别。

与「拐点」侧重结构性事件不同，这里跟踪的是动量方向与加速度的状态变化：

  动量得分 = 以下四票的合计（每票 ±1）：
    1. MACD 柱体符号（多头/空头动能）
    2. MACD 柱体斜率（动能在加速还是衰减）
    3. ROC20 符号（20 日动量方向）
    4. RSI 相对 50（中线强弱）

  得分 ≥ +2 为多头动量区，≤ -2 为空头动量区，中间为过渡区。
  「动量切换」= 得分从一个稳定区进入另一个区，或穿越过渡区完成换向。
"""

from __future__ import annotations

import pandas as pd


def momentum_score(df: pd.DataFrame) -> pd.Series:
    """逐行计算 -4 ~ +4 的动量得分。"""
    hist = df["macd_hist"]
    score = (
        hist.apply(lambda x: 1 if x > 0 else -1)
        + (hist.diff() > 0).astype(int) * 2 - 1
        + df["roc20"].apply(lambda x: 1 if x > 0 else -1)
        + df["rsi14"].apply(lambda x: 1 if x > 50 else -1)
    )
    return score


def momentum_state(df: pd.DataFrame) -> dict:
    """返回当前动量状态与各分量的明细。"""
    row = df.iloc[-1]
    hist, prev_hist = row["macd_hist"], df["macd_hist"].iloc[-2]
    score = int(momentum_score(df).iloc[-1])

    if score >= 2:
        state, label = "bull", "多头动量"
    elif score <= -2:
        state, label = "bear", "空头动量"
    else:
        state, label = "transition", "过渡区（动量换挡中）"

    components = [
        ("MACD柱符号", "多头动能" if hist > 0 else "空头动能", hist > 0),
        ("MACD柱斜率", "动能加速" if hist > prev_hist else "动能衰减", hist > prev_hist),
        ("ROC20", f"20日动量 {'为正' if row['roc20'] > 0 else '为负'} ({row['roc20']:.1f}%)", row["roc20"] > 0),
        ("RSI vs 50", f"RSI {row['rsi14']:.0f} {'强于' if row['rsi14'] > 50 else '弱于'}中线", row["rsi14"] > 50),
    ]
    return {"state": state, "label": label, "score": score, "components": components}


def detect_momentum_switches(df: pd.DataFrame, lookback: int = 30) -> list[dict]:
    """扫描最近 lookback 根内的动量换向事件。

    事件定义：得分从 ≥+2 区域跌入 ≤-2（bull→bear），或反向（bear→bull）。
    使用「最近一次离开稳定区」的状态做比较，过渡区的来回抖动不触发。
    """
    score = momentum_score(df)
    # 把得分映射到三态，并向前填充过渡区，得到「最近一次的稳定阵营」
    zone = score.apply(lambda s: "bull" if s >= 2 else ("bear" if s <= -2 else None))
    camp = zone.ffill()

    events = []
    switches = camp[(camp != camp.shift()) & camp.notna() & camp.shift().notna()]
    n = len(df)
    for ts, new_camp in switches.items():
        if (n - df.index.get_loc(ts)) > lookback:
            continue
        if new_camp == "bull":
            desc = ("动量由空转多：MACD 柱、20 日动量、RSI 中线等多数分量已翻多。"
                    "动量切换初期往往伴随趋势的早期阶段，但也可能是反弹陷阱，"
                    "结合量能（是否放量）与市场状态（ADX）判断质量。")
            direction = "bullish"
        else:
            desc = ("动量由多转空：多数动量分量已翻空。若发生在长期上涨之后、"
                    "且伴随顶背离或放量下跌，是趋势级别回调的常见起点。")
            direction = "bearish"
        events.append({
            "date": ts.date().isoformat() if hasattr(ts, "date") else str(ts),
            "type": f"动量切换 → {'多头' if new_camp == 'bull' else '空头'}",
            "direction": direction, "strength": 3, "description": desc,
        })
    return events
