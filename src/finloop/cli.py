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


def cmd_backtest(args):
    from .backtest import STRATEGIES, get_strategy, run_backtest
    from .data import get_daily
    from .indicators import enrich

    ticker = args.ticker.upper()
    df = enrich(get_daily(ticker, period=args.period))
    if df.empty:
        sys.exit(f"无法获取 {ticker} 的行情数据")
    if args.strategy == "all":
        names = [n for n in STRATEGIES if n != "buy_hold"]
    else:
        names = [args.strategy]
    for name in names:
        positions = get_strategy(name)(df)
        result = run_backtest(df, positions, strategy_name=name,
                              ticker=ticker, cost_bps=args.cost)
        print("\n" + result.summary())
    print(f"\n{DISCLAIMER}")
    print("提醒：单票回测受幸存者偏差与过拟合影响，结论需多票+样本外验证，见 docs/backtesting.md")


def cmd_eventstudy(args):
    from .backtest import event_study
    from .backtest.event_study import format_event_study
    from .data import get_daily
    from .indicators import enrich

    ticker = args.ticker.upper()
    df = enrich(get_daily(ticker, period=args.period))
    if df.empty:
        sys.exit(f"无法获取 {ticker} 的行情数据")
    print(format_event_study(event_study(df), ticker))


def cmd_quality(args):
    from .data.service import default_service

    if args.demo:
        from .data.demo import run_demo
        print(run_demo())
        return
    if not args.ticker:
        sys.exit("用法：finloop quality TICKER 或 finloop quality --demo")
    ticker = args.ticker.upper()
    df = default_service.get_daily(ticker, period="6mo")
    print(f"\n数据源：{default_service.last_source}")
    if default_service.last_report:
        print(default_service.last_report.summary())
    if df.empty:
        sys.exit(1)
    print("\n双源对账（yfinance vs stooq，最近6个月收盘价）：")
    chk = default_service.cross_check(ticker)
    if "error" in chk:
        print(f"  对账失败：{chk['error']}")
    elif chk.get("overlap_days", 0) == 0:
        print("  无重叠数据（备源可能不支持该代码）")
    else:
        print(f"  重叠 {chk['overlap_days']} 天 | 平均偏差 {chk['mean_abs_dev']:.3%} "
              f"| 最大偏差 {chk['max_abs_dev']:.3%} | 超容差(1%) {chk['n_breaches']} 天")
        if chk["n_breaches"]:
            print(f"  超容差日期（前5）：{', '.join(chk['breach_dates'])}")
            print("  注意：两源复权口径不同（yfinance 全调整 vs stooq 仅拆股），分红区间会有系统性偏差。")


def cmd_screen(args):
    from .screener import format_screen, run_screen

    scored = run_screen(universe=args.universe, tier=args.tier)
    if scored.empty:
        sys.exit("无数据（检查网络或 universe 配置）")
    print(format_screen(scored, args.tier))
    if scored.attrs.get("snapshot_path"):
        print(f"\n原始基本面快照已存档：{scored.attrs['snapshot_path']}")
    print(f"\n{DISCLAIMER}")


def cmd_research(args):
    from .backtest.batch import (run_real_matrix, run_scenario_matrix,
                                 write_real_report, write_scenario_report)

    if args.synthetic:
        print(f"运行情景压力测试：{args.paths} 条路径/情景 …")
        matrix = run_scenario_matrix(n_paths=args.paths, cost_bps=args.cost)
        path = write_scenario_report(matrix, cost_bps=args.cost)
        print(f"研究报告已生成：{path}")
        return

    from .config import load_universe, load_watchlist
    if args.tier:
        uni = load_universe("ai")
        tickers = (list(uni.get(args.tier, {})) if args.tier != "all"
                   else [t for tier in uni.values() for t in tier])
    else:
        tickers = load_watchlist()["tickers"]
    print(f"运行横截面回测：{len(tickers)} 只标的 × 全部策略（{args.period}）…")
    matrix, yearly = run_real_matrix(tickers, period=args.period, cost_bps=args.cost)
    if matrix.empty:
        sys.exit("无可用数据（检查网络）")
    path = write_real_report(matrix, yearly, period=args.period, cost_bps=args.cost)
    print(f"研究报告已生成：{path}")


def cmd_loop_status(args):
    from .research_loop import evaluate, format_status, load_state

    st = evaluate(load_state())
    print(format_status(st))
    sys.exit(0 if st.can_exit else 1)  # 退出码可用于自动化（CI/cron 判定）


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

    p = sub.add_parser("backtest", help="策略回测")
    p.add_argument("ticker")
    p.add_argument("--strategy", default="all",
                   help="策略名（buy_hold/sma200/macd/rsi_mr/momentum/all，默认all）")
    p.add_argument("--period", default="5y", help="回测区间（默认5y）")
    p.add_argument("--cost", type=float, default=10.0, help="单边成本bps（默认10）")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("eventstudy", help="信号事件研究：验证信号是否携带信息")
    p.add_argument("ticker")
    p.add_argument("--period", default="5y", help="研究区间（默认5y）")
    p.set_defaults(func=cmd_eventstudy)

    p = sub.add_parser("quality", help="数据质量报告与双源对账")
    p.add_argument("ticker", nargs="?", default=None)
    p.add_argument("--demo", action="store_true",
                   help="离线演示：故障注入→校验判定→降级行为")
    p.set_defaults(func=cmd_quality)

    p = sub.add_parser("screen", help="产业链筛选：需求 vs 估值横截面打分")
    p.add_argument("--universe", default="ai", help="universe 名（默认 ai）")
    p.add_argument("--tier", default="upstream",
                   help="upstream/midstream/downstream/all（默认 upstream）")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("research", help="批量回测研究报告（横截面/情景压力测试）")
    p.add_argument("--synthetic", action="store_true",
                   help="情景压力测试模式（离线，合成多状态行情）")
    p.add_argument("--tier", default=None,
                   help="真实模式：用 AI universe 的某个 tier 替代自选股")
    p.add_argument("--period", default="5y")
    p.add_argument("--cost", type=float, default=10.0)
    p.add_argument("--paths", type=int, default=5, help="情景模式：每情景路径数")
    p.set_defaults(func=cmd_research)

    p = sub.add_parser("loop-status", help="调研循环状态：退出判定与下一轮任务清单")
    p.set_defaults(func=cmd_loop_status)

    p = sub.add_parser("explain", help="指标教学详解")
    p.add_argument("indicator")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("watchlist", help="自选股行情快照")
    p.set_defaults(func=cmd_watchlist)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
