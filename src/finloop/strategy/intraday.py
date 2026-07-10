"""日内交易决策支撑（基于 5 分钟线，默认最近 5 个交易日）。

输出「日内视角」：
  1. VWAP 关系：当日价格相对 VWAP 的位置与偏离度
  2. 开盘区间（Opening Range，前 30 分钟高低点）及突破状态
  3. 日内动量：9/21 EMA 关系、日内 RSI
  4. 关键价位：昨日高低收、当日 VWAP、开盘区间边界
  5. 波动环境：当日已实现波幅 vs 近 5 日平均（节奏快慢）

方法论详见 docs/intraday.md。
"""

from __future__ import annotations

import pandas as pd

from ..indicators import enrich


def intraday_view(df_5m: pd.DataFrame, ticker: str) -> dict:
    """df_5m 为原始 5 分钟 OHLCV（未 enrich），需包含至少 2 个交易日。"""
    if df_5m.empty:
        return {"ticker": ticker, "error": "无日内数据"}

    df = enrich(df_5m, intraday=True)
    days = sorted(set(df.index.date))
    today = days[-1]
    today_df = df[df.index.date == today]
    prev_df = df[df.index.date == days[-2]] if len(days) >= 2 else pd.DataFrame()

    row = today_df.iloc[-1]
    price = float(row["close"])
    out: dict = {"ticker": ticker, "session_date": str(today), "price": price}

    # --- VWAP ---
    vwap_now = float(row["vwap"])
    dev_pct = (price / vwap_now - 1) * 100
    out["vwap"] = {
        "value": round(vwap_now, 2),
        "deviation_pct": round(dev_pct, 2),
        "side": "above" if price > vwap_now else "below",
        "reading": (
            f"价格在 VWAP {'上方' if price > vwap_now else '下方'}，偏离 {dev_pct:+.2f}%。"
            + ("日内买方占优，回踩 VWAP 不破是顺势做多的经典位置。" if price > vwap_now
               else "日内卖方占优，反弹至 VWAP 受阻是顺势做空/离场的观察位。")
            + (" 偏离已较大，注意向 VWAP 回归的引力。" if abs(dev_pct) > 1.5 else "")
        ),
    }

    # --- 开盘区间（当日**首 30 分钟**）---
    # 从数据本身的首根 K 线时间戳推断开盘时刻，而非硬编码美东 09:30——
    # 该框架明确支持台/韩/日(09:00 开盘)等非美市场，硬编码会算错开盘区间。
    session_open = today_df.index[0]
    try:
        or_bars = today_df[today_df.index < session_open + pd.Timedelta(minutes=30)]
    except TypeError:
        or_bars = today_df.iloc[:6]
    if or_bars.empty:
        or_bars = today_df.iloc[:6]
    or_high, or_low = float(or_bars["high"].max()), float(or_bars["low"].min())
    if price > or_high:
        or_state = "已向上突破开盘区间——日内多头格局，区间上沿转为支撑"
    elif price < or_low:
        or_state = "已向下跌破开盘区间——日内空头格局，区间下沿转为压力"
    else:
        or_state = "仍在开盘区间内震荡，方向未选择，突破前观望"
    out["opening_range"] = {"high": round(or_high, 2), "low": round(or_low, 2), "reading": or_state}

    # --- 日内动量 ---
    ema_bull = bool(row["ema9"] > row["ema21"])
    out["momentum"] = {
        "ema9_above_ema21": ema_bull,
        "rsi": round(float(row["rsi14"]), 1),
        "reading": (
            f"9EMA {'在' if ema_bull else '低于'} 21EMA {'上方' if ema_bull else ''}，"
            f"日内 RSI {row['rsi14']:.0f}，"
            + ("短线节奏偏多。" if ema_bull and row["rsi14"] > 50
               else "短线节奏偏空。" if not ema_bull and row["rsi14"] < 50
               else "动量信号混杂，等待一致。")
        ),
    }

    # --- 关键价位 ---
    levels = {"当日VWAP": vwap_now, "开盘区间高点": or_high, "开盘区间低点": or_low}
    if not prev_df.empty:
        levels.update({
            "昨日最高": float(prev_df["high"].max()),
            "昨日最低": float(prev_df["low"].min()),
            "昨日收盘": float(prev_df["close"].iloc[-1]),
        })
    out["key_levels"] = {k: round(v, 2) for k, v in
                         sorted(levels.items(), key=lambda kv: -kv[1])}

    # --- 波动环境 ---
    day_ranges = df.groupby(df.index.date).apply(
        lambda g: (g["high"].max() - g["low"].min()) / g["close"].iloc[-1] * 100
    )
    today_range = float(day_ranges.iloc[-1])
    avg_range = float(day_ranges.iloc[:-1].mean()) if len(day_ranges) > 1 else today_range
    out["volatility"] = {
        "today_range_pct": round(today_range, 2),
        "avg_range_pct": round(avg_range, 2),
        "reading": (
            f"当日振幅 {today_range:.2f}% vs 近几日均值 {avg_range:.2f}%。"
            + ("波动放大，机会与风险同步增加，应缩小仓位、放宽止损至适配波幅。" if today_range > avg_range * 1.3
               else "波动收缩，趋势性机会少，谨防来回扫损。" if today_range < avg_range * 0.7
               else "波动正常。")
        ),
    }
    return out
