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


def detect_turning_points(df: pd.DataFrame, lookback: int = 30,
                          cap: int | None = 10) -> list[dict]:
    """扫描最近 lookback 根 K 线内的拐点事件，按时间排序返回。

    cap：每类信号最多返回最近 cap 次（报告场景防噪声）。
    统计用途（事件研究）必须传 cap=None 取全样本——
    截断成「最近 N 次」会引入近因偏差（审计 M1）。
    """
    events: list[dict] = []
    n = len(df)
    if n < 60:
        return events

    def add(mask: pd.Series, type_: str, direction: str, strength: int, desc: str):
        hits = df.index[mask.fillna(False)]
        if cap is not None:
            hits = hits[-cap:]
        for ts in hits:
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
    """背离检测：在**价格摆动极值处**配对比较指标读数（而非各窗口的指标自身极值）。

    正确定义（修复审计 divergence-method / obv-drift）：
      顶背离 = 近端价格波峰 > 前端价格波峰，但**这两个价格波峰所在位置的**指标读数
              近端 < 前端（指标未跟随价格创新高）。用「价格峰处的指标值」而非
              「指标自身的窗口最大值」——后者会拿两窗口各自最强的一根比，与价格
              峰无关，且对 OBV 这类累计漂移序列几乎恒不触发。
      底背离 = 近端价格波谷 < 前端价格波谷，但两波谷处指标读数近端 > 前端。
    信号日期锚定在**近端摆动极值那一根**（在序列内部，非末根）——使事件研究
    能计算其前向收益（此前锚在末根 → 前向收益恒 NaN → 背离从不进入事件研究）。
    """
    out = []
    if lookback * 2 > len(df) or lookback < 3:
        return out  # 需要两个不重叠窗口；event_study 的 lookback=len(df) 场景优雅跳过
    recent, prior = df.iloc[-lookback:], df.iloc[-lookback * 2:-lookback]

    for col, label in (("rsi14", "RSI"), ("obv", "OBV")):
        # 顶背离：价格峰处配对
        r_hi, p_hi = recent["close"].idxmax(), prior["close"].idxmax()
        if recent.loc[r_hi, "close"] > prior.loc[p_hi, "close"] and \
           recent.loc[r_hi, col] < prior.loc[p_hi, col]:
            out.append({
                "date": r_hi.date().isoformat() if hasattr(r_hi, "date") else str(r_hi),
                "type": f"{label}顶背离", "direction": "bearish", "strength": 2,
                "description": f"价格创出更高的高点，但该高点处的 {label} 读数低于前一波峰：上涨的内在动能/资金支持在减弱，趋势衰竭预警（背离是减仓提示，非精确反转时点）。",
            })
        # 底背离：价格谷处配对
        r_lo, p_lo = recent["close"].idxmin(), prior["close"].idxmin()
        if recent.loc[r_lo, "close"] < prior.loc[p_lo, "close"] and \
           recent.loc[r_lo, col] > prior.loc[p_lo, col]:
            out.append({
                "date": r_lo.date().isoformat() if hasattr(r_lo, "date") else str(r_lo),
                "type": f"{label}底背离", "direction": "bullish", "strength": 2,
                "description": f"价格创出更低的低点，但该低点处的 {label} 读数高于前一波谷：抛压在衰竭，下跌动能与价格背离，关注企稳反转的可能。",
            })
    return out
