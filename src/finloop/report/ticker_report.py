"""个股完整分析报告（Markdown）：指标解读 + 市场状态 + 信号 + 长线视角。"""

from __future__ import annotations

from .. import DISCLAIMER
from ..data import get_daily, get_news
from ..indicators import enrich
from ..indicators.explain import explain_snapshot
from ..signals import (classify_regime, detect_momentum_switches,
                       detect_turning_points, momentum_state)
from ..strategy import long_term_view

STRENGTH_MARK = {1: "△", 2: "▲", 3: "★"}


def _fmt_signals(events: list[dict]) -> list[str]:
    lines = []
    for e in events:
        arrow = "🟢" if e["direction"] == "bullish" else "🔴"
        mark = STRENGTH_MARK.get(e["strength"], "")
        lines.append(f"- {arrow} **{e['date']} {e['type']}** {mark}\n  {e['description']}")
    return lines or ["- 最近观察窗口内无显著信号。"]


def build_ticker_report(ticker: str, with_fundamentals: bool = True) -> str:
    df = enrich(get_daily(ticker, period="2y"))
    if df.empty:
        return f"# {ticker}\n\n无法获取行情数据。"

    row = df.iloc[-1]
    chg = (row["close"] / df["close"].iloc[-2] - 1) * 100
    lt = long_term_view(df, ticker, with_fundamentals=with_fundamentals)
    regime = classify_regime(df)
    mom = momentum_state(df)

    lines = [
        f"# {lt.get('name', ticker)} ({ticker}) 分析报告",
        "",
        f"**日期** {df.index[-1].date()} | **收盘** {row['close']:.2f}（{chg:+.2f}%）"
        + (f" | **行业** {lt['sector']}" if lt.get("sector") else ""),
        "",
        f"## 市场状态：{regime['label']}",
        "",
        *(f"- {e}" for e in regime["evidence"]),
        "",
        f"> **策略含义**：{regime['hint']}",
        "",
        f"## 动量状态：{mom['label']}（得分 {mom['score']:+d}/4）",
        "",
    ]
    for name, desc, bullish in mom["components"]:
        lines.append(f"- {'🟢' if bullish else '🔴'} {name}：{desc}")

    lines += ["", "## 指标详解", ""]
    for item in explain_snapshot(df):
        lines += [
            f"### {item['indicator']} — {item['reading']}",
            f"`{item['value']}`",
            "",
            item["detail"],
            "",
        ]

    lines += ["## 近 30 日拐点与动量切换信号", ""]
    events = detect_turning_points(df) + detect_momentum_switches(df)
    events.sort(key=lambda e: e["date"])
    lines += _fmt_signals(events)

    # 长线视角
    lines += ["", f"## 长线视角（检查清单得分 {lt['score']}）", ""]
    t = lt["trend"]
    lines += [
        f"- 距 52 周高点：{t['off_52w_high_pct']}% | 距 52 周低点：+{t['off_52w_low_pct']}%"
        + (f" | 近一年涨幅：{t['ret_1y_pct']}%" if t["ret_1y_pct"] is not None else ""),
        "",
    ]
    for c in lt["checklist"]:
        lines.append(f"- {c['status']} {c['item']}\n  {c['note']}")
    lines += ["", "  ⚠️ 检查清单阈值（ROE>15%/净利率>10%/PEG<2）是**行业无差别通用值**，"
              "对分销/EMS/公用事业等结构性低利润率行业会系统性偏严——需结合"
              f"行业（{lt.get('sector') or '未知'}）解读，勿机械按 ✅/❌ 计数。"]
    if lt.get("data_warnings"):
        lines += ["", "### ⚠️ 基本面数据预警（采信前先人工核验）", ""]
        lines += [f"- {w}" for w in lt["data_warnings"]]
    if lt["fundamentals"]:
        lines += ["", "### 基本面快照", "", "| 指标 | 数值 | 说明 |", "|---|---|---|"]
        for v in lt["fundamentals"].values():
            lines.append(f"| {v['label']} | {v['formatted']} | {v['note']} |")
    lines += ["", f"> **操作参考**：{lt['action_note']}"]

    news = get_news(ticker, limit=5)
    if news:
        lines += ["", "## 相关新闻", ""]
        for n in news:
            lines.append(f"- [{n['title']}]({n['link']})（{n['publisher']}）")

    lines += ["", "---", DISCLAIMER, ""]
    return "\n".join(lines)
