"""finloop 命令行入口。

  finloop report              生成今日市场日报（写入 reports/）
  finloop analyze TICKER      个股完整分析报告
  finloop signals TICKER      只看拐点与动量切换信号
  finloop intraday TICKER     日内视角（5 分钟线）
  finloop explain INDICATOR   指标教学详解（rsi/macd/bollinger/...）
  finloop watchlist           查看自选股行情快照
"""

from __future__ import annotations

import argparse
import sys

from . import DISCLAIMER, __version__


def cmd_report(args):
    from .report import build_daily_report
    from .config import reports_dir
    import datetime as dt

    report = build_daily_report(write=not args.stdout)
    if args.stdout:
        print(report)
    else:
        path = reports_dir() / f"{dt.date.today().isoformat()}.md"
        print(f"日报已生成：{path}")


def cmd_analyze(args):
    from .report import build_ticker_report
    print(build_ticker_report(args.ticker.upper(), with_fundamentals=not args.no_fundamentals))


def cmd_signals(args):
    from .data import get_daily
    from .indicators import enrich
    from .signals import detect_momentum_switches, detect_turning_points, momentum_state

    df = enrich(get_daily(args.ticker.upper(), period="2y"))
    if df.empty:
        sys.exit(f"无法获取 {args.ticker} 的行情数据")
    mom = momentum_state(df)
    print(f"\n{args.ticker.upper()} 当前动量状态：{mom['label']}（得分 {mom['score']:+d}/4）\n")
    events = detect_turning_points(df, lookback=args.days) + \
        detect_momentum_switches(df, lookback=args.days)
    events.sort(key=lambda e: e["date"])
    if not events:
        print(f"最近 {args.days} 个交易日内无显著信号。")
    for e in events:
        arrow = "🟢" if e["direction"] == "bullish" else "🔴"
        print(f"{arrow} [{e['date']}] {e['type']}（强度{e['strength']}）")
        print(f"   {e['description']}\n")
    print(DISCLAIMER)


def cmd_intraday(args):
    from .data import get_intraday
    from .strategy import intraday_view

    view = intraday_view(get_intraday(args.ticker.upper(), interval="5m", period="5d"),
                         args.ticker.upper())
    if "error" in view:
        sys.exit(view["error"])
    print(f"\n{view['ticker']} 日内视角（{view['session_date']}）  现价 {view['price']:.2f}\n")
    print(f"◆ VWAP：{view['vwap']['value']}\n  {view['vwap']['reading']}\n")
    o = view["opening_range"]
    print(f"◆ 开盘区间(前30分钟)：{o['low']} ~ {o['high']}\n  {o['reading']}\n")
    print(f"◆ 日内动量：{view['momentum']['reading']}\n")
    print("◆ 关键价位（由高到低）：")
    for name, level in view["key_levels"].items():
        marker = " ← 现价附近" if abs(level / view["price"] - 1) < 0.002 else ""
        print(f"   {name:<8} {level}{marker}")
    print(f"\n◆ 波动环境：{view['volatility']['reading']}\n")
    print(DISCLAIMER)


def cmd_explain(args):
    from .indicators.explain import LESSONS, format_lesson, get_lesson

    lesson = get_lesson(args.indicator)
    if lesson is None:
        print(f"未找到指标 '{args.indicator}'。可用：{', '.join(sorted(LESSONS))}")
        sys.exit(1)
    print(format_lesson(lesson))


def cmd_watchlist(args):
    from .config import load_watchlist
    from .data import get_last_quote

    cfg = load_watchlist()
    print(f"\n自选股（config/watchlist.yaml）：\n")
    for t in cfg["tickers"]:
        q = get_last_quote(t)
        if q["price"] is None:
            print(f"  {t:<6} 数据获取失败")
        else:
            emoji = "🟢" if q["change_pct"] >= 0 else "🔴"
            print(f"  {t:<6} {q['price']:>10.2f}  {emoji} {q['change_pct']:+.2f}%  ({q['date']})")
    print()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="finloop",
        description="美股金融分析与投资决策框架（学习用途，非投资建议）",
    )
    parser.add_argument("--version", action="version", version=f"finloop {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("report", help="生成今日市场日报")
    p.add_argument("--stdout", action="store_true", help="打印到终端而不写文件")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("analyze", help="个股完整分析报告")
    p.add_argument("ticker")
    p.add_argument("--no-fundamentals", action="store_true", help="跳过基本面（更快）")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("signals", help="拐点与动量切换信号")
    p.add_argument("ticker")
    p.add_argument("--days", type=int, default=30, help="回看交易日数（默认30）")
    p.set_defaults(func=cmd_signals)

    p = sub.add_parser("intraday", help="日内视角分析（5分钟线）")
    p.add_argument("ticker")
    p.set_defaults(func=cmd_intraday)

    p = sub.add_parser("explain", help="指标教学详解")
    p.add_argument("indicator")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("watchlist", help="自选股行情快照")
    p.set_defaults(func=cmd_watchlist)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
