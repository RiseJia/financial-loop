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
