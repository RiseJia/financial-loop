# Financial Loop — 美股分析与投资决策框架

一个面向美股的个人金融分析与学习框架，提供**长线投资**与**日内交易**两条决策支撑线，
并能每天自动收集市场关键信息、生成结构化的 Markdown 日报。

> ⚠️ **免责声明**：本项目仅用于金融学习与研究，所有输出均为程序化分析结果，
> 不构成任何投资建议。投资有风险，决策需独立判断。

## 核心功能

| 模块 | 功能 |
|------|------|
| `finloop.data` | 多源数据层：yfinance 主源 + stooq 备源降级、质量校验管道、parquet 本地缓存、双源对账 |
| `finloop.backtest` | 回测引擎：策略抽象、向量化引擎（防前视+成本模型）、绩效指标、信号事件研究 |
| `finloop.indicators` | 趋势 / 动量 / 波动率 / 量能四大类技术指标，纯 pandas 实现，无 TA-Lib 依赖 |
| `finloop.indicators.explain` | 指标解释引擎：对每个指标的当前读数生成中文详细解读 |
| `finloop.signals` | 拐点检测（金叉死叉、背离、布林挤压突破等）与动量切换（momentum switching）识别 |
| `finloop.signals.regime` | 市场状态机：趋势市 / 震荡市 / 转折期 分类 |
| `finloop.strategy.long_term` | 长线决策支撑：趋势过滤、估值快照、基本面评分、定投参考 |
| `finloop.strategy.intraday` | 日内决策支撑：VWAP、开盘区间、日内动量、关键价位 |
| `finloop.report` | 每日市场报告：大盘概览、板块轮动、宏观代理指标、自选股信号、新闻摘要 |

## 快速开始

```bash
pip install -r requirements.txt
pip install -e .

# 生成今日市场日报（输出到 reports/YYYY-MM-DD.md）
finloop report

# 对单只股票做完整分析（指标 + 解读 + 信号 + 长线视角）
finloop analyze AAPL

# 只看拐点与动量切换信号
finloop signals NVDA

# 日内视角分析（基于 5 分钟线）
finloop intraday TSLA

# 查看某个指标的详细中文教学解释
finloop explain rsi
finloop explain macd

# 回测：策略 vs 买入持有（5年，含交易成本）
finloop backtest AAPL --strategy sma200
finloop backtest SPY --strategy all

# 信号事件研究：验证拐点/动量信号是否真的携带信息
finloop eventstudy NVDA

# 调研循环：退出判定与下一轮任务清单（状态在 config/research_state.yaml）
finloop loop-status

# 数据质量：真实检查 / 离线故障注入演示
finloop quality NVDA
finloop quality --demo

# AI 产业链筛选：需求 vs 估值横截面打分（universe 见 config/universe_ai.yaml）
finloop screen --tier upstream
finloop screen --tier all

# 研究级批量回测：横截面（多标的）/ 情景压力测试（离线）
finloop research --tier upstream --period 5y
finloop research --synthetic

# 查看自选股列表
finloop watchlist
```

自选股配置在 `config/watchlist.yaml`，可自由增删。

## 目录结构

```
src/finloop/
├── data/          # 数据层：行情、基本面、新闻
├── indicators/    # 指标层：trend / momentum / volatility / volume + 解释引擎
├── signals/       # 信号层：拐点、动量切换、市场状态
├── strategy/      # 策略层：长线 / 日内决策支撑
└── report/        # 报告层：每日市场报告、个股报告
docs/              # 方法论文档（指标详解、拐点理论、长线与日内框架）
reports/           # 每日生成的报告归档
tests/             # 指标与信号的单元测试（离线合成数据）
```

## 方法论文档

**先读这两篇**：[docs/workflow.md](docs/workflow.md)（工作流总纲：五部件如何咬合）→
[docs/chain/](docs/chain/README.md)（AI 产业链体系化教程，8 篇）。

- [docs/indicators.md](docs/indicators.md) — 每个指标的原理、公式、参数、解读方式与常见误区
- [docs/turning_points.md](docs/turning_points.md) — 拐点与动量切换的识别逻辑
- [docs/long_term.md](docs/long_term.md) — 长线投资决策框架
- [docs/intraday.md](docs/intraday.md) — 日内交易决策框架
- [docs/backtesting.md](docs/backtesting.md) — 回测原理：策略的抽象、引擎会计、四大陷阱、事件研究
- [docs/data_quality.md](docs/data_quality.md) — 数据质量解决方案：校验管线、降级、双源对账、诚实边界
- [docs/research_loop.md](docs/research_loop.md) — 调研循环：退出准则的设计与自动化（loop-status）
- [docs/ai_supply_chain.md](docs/ai_supply_chain.md) — AI 产业链方法论：上游「需求 >> 估值」的三层证据与风险排查
- [docs/ai_supply_chain_map.md](docs/ai_supply_chain_map.md) — 产业链深度地图：细化到材料/元件级（ABF膜→T-glass、EML→InP衬底、变压器→电气钢），含瓶颈评分与需求证据

## 每日自动报告

`.github/workflows/daily-report.yml` 配置了 GitHub Actions 定时任务，
在每个交易日美东收盘后自动运行 `finloop report` 并把日报提交到 `reports/` 目录。
如不需要可直接删除该 workflow 文件。

## 数据来源说明

默认使用 [yfinance](https://github.com/ranaroussi/yfinance)（雅虎财经免费数据）：

- 日线数据：完整历史
- 分钟线：1m 仅最近 7 天，5m 最近 60 天（雅虎限制）
- 基本面：估值、利润率、增长等快照字段
- 新闻：个股与大盘相关头条

数据层做了统一抽象（`finloop.data.market`），后续可平滑替换为 Polygon、Alpha Vantage 等付费源。
