"""拐点检测：在 enrich 后的 DataFrame 上扫描结构性转折事件。

覆盖的拐点类型（详细理论见 docs/turning_points.md）：
  - 均线金叉/死叉（50/200 = 黄金交叉/死亡交叉；20/50 = 中期交叉）
  - MACD 金叉/死叉与零轴穿越
  - RSI 极端区回归（超卖回升穿 30 / 超买回落穿 70）
  - 布林挤压后的放量突破
  - 价格-RSI 背离、价格-OBV 背离
  - 200 日线得失（长线牛熊分界）

每个事件返回 {date, type, direction, strength, description}。
direction: bullish/bearish；strength: 1(弱) ~ 3(强)。
"""

from __future__ import annotations

import pandas as pd


def _cross_up(a: pd.Series, b) -> pd.Series:
    """a 上穿 b（b 可为 Series 或常数）。"""
    return (a > b) & (a.shift() <= (b.shift() if isinstance(b, pd.Series) else b))


def _cross_down(a: pd.Series, b) -> pd.Series:
    return (a < b) & (a.shift() >= (b.shift() if isinstance(b, pd.Series) else b))


def detect_turning_points(df: pd.DataFrame, lookback: int = 30) -> list[dict]:
    """扫描最近 lookback 根 K 线内的拐点事件，按时间排序返回。"""
    events: list[dict] = []
    n = len(df)
    if n < 60:
        return events

    def add(mask: pd.Series, type_: str, direction: str, strength: int, desc: str):
        for ts in df.index[mask.fillna(False)][-10:]:
            if (n - df.index.get_loc(ts)) <= lookback:
                events.append({
                    "date": ts.date().isoformat() if hasattr(ts, "date") else str(ts),
                    "type": type_, "direction": direction,
                    "strength": strength, "description": desc,
                })

    c = df["close"]

    # --- 均线交叉 ---
    if df["sma200"].notna().any():
        add(_cross_up(df["sma50"], df["sma200"]), "黄金交叉", "bullish", 3,
            "50 日均线上穿 200 日均线，历史上是长线牛市结构确立的标志性事件，但信号滞后、确认期长。")
        add(_cross_down(df["sma50"], df["sma200"]), "死亡交叉", "bearish", 3,
            "50 日均线下穿 200 日均线，长线趋势恶化的标志性预警，通常意味着应降低多头敞口。")
        add(_cross_up(c, df["sma200"]), "收复200日线", "bullish", 2,
            "价格重新站上 200 日均线，长线牛熊分界线得而复失/失而复得的关键观察点，需看后续能否站稳。")
        add(_cross_down(c, df["sma200"]), "跌破200日线", "bearish", 2,
            "价格跌破 200 日均线，长线趋势的第一道防线告破。")
    add(_cross_up(df["sma20"], df["sma50"]), "中期均线金叉", "bullish", 2,
        "20 日均线上穿 50 日均线，中期趋势转多的确认信号。")
    add(_cross_down(df["sma20"], df["sma50"]), "中期均线死叉", "bearish", 2,
        "20 日均线下穿 50 日均线，中期趋势转弱。")

    # --- MACD ---
    macd_, sig = df["macd"], df["macd_signal"]
    above_zero = macd_ > 0
    gc = _cross_up(macd_, sig)
    dc = _cross_down(macd_, sig)
    add(gc & above_zero, "MACD零上金叉", "bullish", 3,
        "MACD 在零轴上方金叉：上升趋势中的回调结束信号，是金叉中可靠度最高的一类。")
    add(gc & ~above_zero, "MACD零下金叉", "bullish", 1,
        "MACD 在零轴下方金叉：下跌中的反弹尝试，可靠度有限，需其他证据配合。")
    add(dc & above_zero, "MACD零上死叉", "bearish", 1,
        "MACD 在零轴上方死叉：上涨途中的动能减弱，可能只是休整。")
    add(dc & ~above_zero, "MACD零下死叉", "bearish", 3,
        "MACD 在零轴下方死叉：下跌趋势中的再度转弱，杀伤力最大的一类死叉。")
    add(_cross_up(macd_, 0), "MACD上穿零轴", "bullish", 2,
        "MACD 升至零轴上方，意味着 12 日 EMA 重新高于 26 日 EMA，多头正式接管中期动能。")
    add(_cross_down(macd_, 0), "MACD下穿零轴", "bearish", 2,
        "MACD 跌至零轴下方，空头接管中期动能。")

    # --- RSI 极端回归 ---
    r = df["rsi14"]
    add(_cross_up(r, 30), "RSI超卖回升", "bullish", 2,
        "RSI 从超卖区回升穿越 30：恐慌抛售衰竭后的修复信号，比「正处于超卖」更有操作意义。")
    add(_cross_down(r, 70), "RSI超买回落", "bearish", 2,
        "RSI 从超买区回落穿越 70：过热动能开始降温，强势股可能只是休整，弱势反弹股则警惕见顶。")

    # --- 布林挤压突破 ---
    if n >= 130:
        squeeze = df["bb_width"] < df["bb_width"].rolling(120).quantile(0.15)
        was_squeezed = squeeze.shift().rolling(5).max() > 0
        vol_ok = df["vol_ratio"] > 1.5
        add(_cross_up(c, df["bb_upper"]) & was_squeezed & vol_ok,
            "挤压放量上破", "bullish", 3,
            "布林带宽收缩至近半年低位后放量突破上轨：波动率压缩→释放，新一轮上升行情的高质量起点形态。")
        add(_cross_down(c, df["bb_lower"]) & was_squeezed & vol_ok,
            "挤压放量下破", "bearish", 3,
            "布林挤压后放量跌破下轨：向下选择方向，往往是一段趋势性下跌的开端。")

    # --- 背离（最近 60 根内的高低点比较）---
    events.extend(_detect_divergence(df, lookback))

    events.sort(key=lambda e: e["date"])
    return events


def _detect_divergence(df: pd.DataFrame, lookback: int) -> list[dict]:
    """简化背离检测：比较最近窗口与前一窗口的价格/指标极值。

    顶背离：价格创出更高高点，而 RSI/OBV 的对应高点降低。
    底背离：价格创出更低低点，而 RSI/OBV 的对应低点抬高。
    """
    out = []
    if len(df) < 120:
        return out
    recent, prior = df.iloc[-lookback:], df.iloc[-lookback * 2:-lookback]
    last_date = df.index[-1].date().isoformat()

    for col, label in (("rsi14", "RSI"), ("obv", "OBV")):
        # 顶背离
        if recent["close"].max() > prior["close"].max() and \
           recent[col].max() < prior[col].max():
            out.append({
                "date": last_date, "type": f"{label}顶背离", "direction": "bearish",
                "strength": 2,
                "description": f"价格创出新高但 {label} 高点降低：上涨的内在动能/资金支持在减弱，趋势衰竭预警（背离不是反转的精确时点信号，而是减仓提示）。",
            })
        # 底背离
        if recent["close"].min() < prior["close"].min() and \
           recent[col].min() > prior[col].min():
            out.append({
                "date": last_date, "type": f"{label}底背离", "direction": "bullish",
                "strength": 2,
                "description": f"价格创出新低但 {label} 低点抬高：抛压在衰竭，下跌动能与价格走势背离，关注企稳反转的可能。",
            })
    return out
