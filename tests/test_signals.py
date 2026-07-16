from finloop.indicators import enrich
from finloop.indicators.explain import (LESSONS, explain_snapshot,
                                        format_lesson, get_lesson)
from finloop.signals import (classify_regime, detect_momentum_switches,
                             detect_turning_points, momentum_state)


def test_regime_trend_up(trending_up):
    r = classify_regime(enrich(trending_up))
    assert r["regime"] == "trend_up"
    assert r["hint"]


def test_regime_trend_down(trending_down):
    r = classify_regime(enrich(trending_down))
    assert r["regime"] == "trend_down"


def test_momentum_state_bull():
    # 用确定性的指数增长序列，保证所有动量分量同向为多
    import numpy as np
    from tests.conftest import make_ohlcv

    df = enrich(make_ohlcv(100 * 1.003 ** np.arange(300)))
    m = momentum_state(df)
    assert m["state"] == "bull"
    assert m["score"] >= 2
    assert len(m["components"]) == 4


def test_momentum_state_bear(trending_down):
    assert momentum_state(enrich(trending_down))["state"] == "bear"


def test_v_reversal_triggers_bullish_switch(v_reversal):
    df = enrich(v_reversal)
    # V 型反转后半段应出现「动量切换 → 多头」事件
    events = detect_momentum_switches(df, lookback=200)
    assert any(e["direction"] == "bullish" for e in events)


def test_v_reversal_turning_points(v_reversal):
    df = enrich(v_reversal)
    events = detect_turning_points(df, lookback=200)
    types = {e["type"] for e in events}
    # 反转过程必然出现 MACD 上穿零轴与收复 200 日线之一
    assert types & {"MACD上穿零轴", "收复200日线", "中期均线金叉", "黄金交叉"}
    for e in events:
        assert e["direction"] in ("bullish", "bearish")
        assert e["strength"] in (1, 2, 3)
        assert e["description"]


def test_explain_snapshot_structure(trending_up):
    items = explain_snapshot(enrich(trending_up))
    assert len(items) >= 5
    for it in items:
        assert it["indicator"] and it["reading"] and it["detail"]


def test_lessons_complete():
    for key, lesson in LESSONS.items():
        for field in ("name", "category", "formula", "principle", "usage", "pitfalls"):
            assert lesson[field], f"{key}.{field} 为空"
    assert get_lesson("RSI")["name"].startswith("RSI")
    assert get_lesson("kd") is not None  # 别名
    assert "公式" in format_lesson(get_lesson("macd"))


# ---------------------------------------------------------------- 对抗审计：信号/策略修复

def test_regime_hysteresis_no_flipflop():
    """单根异常状态不改变已确认 regime——消除逐日抖动。"""
    from finloop.signals import classify_regime
    df = enrich(trending_up_fixture())
    base = classify_regime(df)["regime"]
    # 篡改最后一根的 adx 到灰色地带（单根），不应立即翻转已确认状态
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("adx")] = 10.0
    flipped = classify_regime(df2)["regime"]
    assert flipped == base   # 单根扰动被滞回吸收


def trending_up_fixture():
    import numpy as np
    from tests.conftest import make_ohlcv
    rng = np.random.default_rng(42)
    return make_ohlcv(100 * np.cumprod(1 + 0.003 + rng.normal(0, 0.005, 300)))


def test_long_term_peg_negative_not_pass():
    """负 PEG 不应被判 ✅（PEG 对亏损/负增长失效）。"""
    import pandas as pd, numpy as np
    from unittest.mock import patch
    from finloop.strategy.long_term import long_term_view
    from tests.conftest import make_ohlcv
    df = enrich(make_ohlcv(100 * 1.003 ** np.arange(300)))
    fake = {"pegRatio": {"label": "PEG", "value": -1.5, "note": ""},
            "_name": "X", "_sector": "Tech"}
    with patch("finloop.strategy.long_term.get_fundamentals", return_value=fake):
        v = long_term_view(df, "X")
    peg_check = [c for c in v["checklist"] if "PEG" in c["item"]][0]
    assert peg_check["status"] == "❌"   # 负 PEG 判不通过，不再误判 ✅


def test_divergence_paired_at_swing_and_dated_internally():
    """背离在价格摆动极值处配对，且信号日期在序列内部（事件研究可用）。"""
    import numpy as np
    from finloop.signals.turning_points import _detect_divergence
    from tests.conftest import make_ohlcv
    p1 = np.concatenate([np.linspace(80, 100, 30), np.linspace(100, 90, 30)])
    p2 = np.concatenate([np.linspace(90, 105, 40), np.linspace(105, 100, 20)])
    df = enrich(make_ohlcv(np.concatenate([p1, p2])))
    divs = _detect_divergence(df, lookback=60)
    assert any(d["type"] == "RSI顶背离" for d in divs)     # 价格更高峰+RSI更低峰
    last = df.index[-1].date().isoformat()
    assert all(d["date"] != last for d in divs)            # 锚在摆动点，非末根


def test_divergence_empty_when_windows_overlap():
    """event_study 传 lookback=len(df) 时两窗口无法不重叠 → 优雅返回空，不报错。"""
    import numpy as np
    from finloop.signals.turning_points import _detect_divergence
    from tests.conftest import make_ohlcv
    df = enrich(make_ohlcv(100 * 1.001 ** np.arange(150)))
    assert _detect_divergence(df, lookback=len(df)) == []


def test_divergence_full_history_scan_for_event_study():
    """行为回归：event_study 语义（lookback=len(df)）下背离产出全历史样本。

    此前实现把 lookback 当窗口宽度，event_study 场景两窗口无法不重叠 →
    背离恒空、从未被统计验证——由行为回测抓出。
    """
    import numpy as np
    from finloop.signals.turning_points import _detect_divergence
    from tests.conftest import make_ohlcv
    # 多段涨跌交替（背离高发形态）
    segs = []
    lvl = 100.0
    rng = np.random.default_rng(3)
    for k in range(6):
        tgt = lvl * (1.25 if k % 2 == 0 else 0.9)
        segs.append(np.linspace(lvl, tgt, 60))
        lvl = tgt
    df = enrich(make_ohlcv(np.concatenate(segs)))
    divs = _detect_divergence(df, lookback=len(df))
    assert len(divs) >= 1                       # 全历史扫描有产出
    # 去重有效：同类型信号锚点间隔 ≥ 半窗
    from finloop.signals.turning_points import DIVERGENCE_WINDOW
    by_type = {}
    for d in divs:
        by_type.setdefault(d["type"], []).append(d["date"])
    idx = {ts.date().isoformat(): i for i, ts in enumerate(df.index)}
    for dates in by_type.values():
        pos = sorted(idx[d] for d in dates)
        assert all(b - a >= DIVERGENCE_WINDOW // 2 for a, b in zip(pos, pos[1:]))


def test_regime_state_machine_reduces_flips_materially():
    """滞回状态机在缠绕序列上把翻转次数至少降 40%（行为标定 persist=5→~52%）。"""
    import numpy as np
    from finloop.signals.regime import _raw_regime, classify_regime
    from tests.conftest import make_ohlcv
    rng = np.random.default_rng(7)
    df = enrich(make_ohlcv(100 * np.cumprod(1 + rng.normal(0.0002, 0.012, 500))))
    warm = int(df["sma200"].isna().sum())
    raws = [_raw_regime(df.iloc[i]) for i in range(warm, len(df))]
    raw_flips = sum(a != b for a, b in zip(raws, raws[1:]))
    conf = [classify_regime(df.iloc[:end])["regime"]
            for end in range(warm + 20, len(df))]
    conf_flips = sum(a != b for a, b in zip(conf, conf[1:]))
    assert conf_flips <= raw_flips * 0.6
