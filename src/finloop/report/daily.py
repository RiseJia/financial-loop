"""每日市场报告：大盘概览、宏观代理、板块轮动、自选股信号、新闻头条。

输出 Markdown，默认写入 reports/YYYY-MM-DD.md。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .. import DISCLAIMER
from ..config import load_watchlist, reports_dir
from ..data import get_daily, get_news
from ..indicators import enrich
from ..signals import (classify_regime, detect_momentum_switches,
                       detect_turning_points, momentum_state)


def _quote_block(tickers: dict[str, str]) -> list[str]:
    """行情表：收盘、日涨跌、5日、20日涨跌。"""
    lines = ["| 标的 | 名称 | 收盘 | 1日 | 5日 | 20日 |", "|---|---|---|---|---|---|"]
    for symbol, name in tickers.items():
        try:
            df = get_daily(symbol, period="3mo")
            if len(df) < 21:
                continue
            c = df["close"]
            r1 = (c.iloc[-1] / c.iloc[-2] - 1) * 100
            r5 = (c.iloc[-1] / c.iloc[-6] - 1) * 100
            r20 = (c.iloc[-1] / c.iloc[-21] - 1) * 100
            emoji = "🟢" if r1 >= 0 else "🔴"
            lines.append(
                f"| {symbol} | {name} | {c.iloc[-1]:.2f} | {emoji} {r1:+.2f}% | {r5:+.2f}% | {r20:+.2f}% |"
            )
        except Exception:
            lines.append(f"| {symbol} | {name} | 数据获取失败 | - | - | - |")
    return lines


def _vix_comment(vix: float) -> str:
    if vix < 15:
        return "市场情绪平静（自满区），波动率成本低，但需警惕低波动环境的突然逆转。"
    if vix < 20:
        return "情绪正常偏稳。"
    if vix < 30:
        return "市场进入紧张状态，避险情绪升温，仓位管理优先于进攻。"
    return "恐慌区！历史上 VIX>30 常伴随阶段性底部的酝酿，但过程往往剧烈。"


def build_daily_report(write: bool = True) -> str:
    cfg = load_watchlist()
    today = dt.date.today().isoformat()
    lines = [f"# 每日市场报告 — {today}", ""]

    # --- 大盘指数 ---
    lines += ["## 一、大盘概览", ""]
    lines += _quote_block(cfg["benchmarks"])

    # 大盘状态：以 SPY 为锚
    try:
        spy = enrich(get_daily("SPY", period="2y"))
        regime = classify_regime(spy)
        mom = momentum_state(spy)
        lines += [
            "",
            f"**市场状态（以 SPY 为锚）**：{regime['label']} | 动量：{mom['label']}（{mom['score']:+d}/4）",
            "",
            *(f"- {e}" for e in regime["evidence"]),
            "",
            f"> {regime['hint']}",
        ]
    except Exception:
        lines += ["", "（SPY 状态分析失败）"]

    # --- 宏观代理 ---
    lines += ["", "## 二、宏观与情绪指标", ""]
    lines += _quote_block(cfg["macro"])
    try:
        vix_df = get_daily("^VIX", period="1mo")
        vix = float(vix_df["close"].iloc[-1])
        lines += ["", f"**VIX = {vix:.1f}** — {_vix_comment(vix)}"]
    except Exception:
        pass
    lines += [
        "",
        "> 解读提示：10 年期美债收益率（^TNX）上行通常压制成长股估值；美元走强利空大宗与海外营收占比高的公司；"
        "VIX 与股指通常负相关，是情绪的温度计。",
    ]

    # --- 板块轮动 ---
    if cfg.get("sectors"):
        lines += ["", "## 三、板块轮动", ""]
        rows = []
        for symbol, name in cfg["sectors"].items():
            try:
                df = get_daily(symbol, period="3mo")
                c = df["close"]
                rows.append((symbol, name,
                             (c.iloc[-1] / c.iloc[-6] - 1) * 100,
                             (c.iloc[-1] / c.iloc[-21] - 1) * 100))
            except Exception:
                continue
        rows.sort(key=lambda r: -r[3])
        lines += ["| 板块 | 名称 | 5日 | 20日 |", "|---|---|---|---|"]
        for symbol, name, r5, r20 in rows:
            lines.append(f"| {symbol} | {name} | {r5:+.2f}% | {r20:+.2f}% |")
        if rows:
            strongest, weakest = rows[0], rows[-1]
            lines += [
                "",
                f"> 近 20 日最强板块：**{strongest[1]} ({strongest[0]})** {strongest[3]:+.1f}%；"
                f"最弱：**{weakest[1]} ({weakest[0]})** {weakest[3]:+.1f}%。"
                "防御板块（必需消费/公用事业）领涨而进攻板块（科技/可选消费）落后时，常是市场转向避险的信号。",
            ]

    # --- 自选股信号扫描 ---
    lines += ["", "## 四、自选股信号扫描", ""]
    for ticker in cfg["tickers"]:
        try:
            df = enrich(get_daily(ticker, period="2y"))
            if df.empty:
                continue
            row = df.iloc[-1]
            chg = (row["close"] / df["close"].iloc[-2] - 1) * 100
            mom = momentum_state(df)
            events = detect_turning_points(df, lookback=5) + detect_momentum_switches(df, lookback=5)
            events.sort(key=lambda e: e["date"])
            lines += [f"### {ticker} — {row['close']:.2f}（{chg:+.2f}%）｜{mom['label']}", ""]
            if events:
                for e in events:
                    arrow = "🟢" if e["direction"] == "bullish" else "🔴"
                    lines.append(f"- {arrow} **{e['date']} {e['type']}**：{e['description']}")
            else:
                lines.append("- 近 5 个交易日无新信号。")
            lines.append("")
        except Exception as exc:
            lines += [f"### {ticker}", f"- 分析失败：{exc}", ""]

    # --- 新闻头条 ---
    lines += ["## 五、市场新闻头条", ""]
    seen = set()
    for src in ["SPY", "QQQ"] + cfg["tickers"][:5]:
        for n in get_news(src, limit=3):
            if n["title"] in seen:
                continue
            seen.add(n["title"])
            lines.append(f"- [{n['title']}]({n['link']})（{n['publisher']}）")
    if not seen:
        lines.append("- 新闻获取失败或无数据。")

    lines += ["", "---", DISCLAIMER, ""]
    report = "\n".join(lines)

    if write:
        path = reports_dir() / f"{today}.md"
        path.write_text(report, encoding="utf-8")
    return report
