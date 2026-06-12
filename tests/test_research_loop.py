import datetime as dt

from finloop.research_loop import evaluate, format_status, load_state

TODAY = dt.date(2026, 6, 15)


def make_state(**overrides):
    base = {
        "meta": {"cadence_days": 90},
        "global_triggers": [
            {"condition": "总闸门", "status": "ok", "last_checked": "2026-06-11"},
        ],
        "nodes": {
            "n1": {
                "label": "节点一",
                "evidence": [{"claim": "x", "asof": "2026-06-11"}],
                "triggers": [{"condition": "c1", "status": "ok",
                              "last_checked": "2026-06-11"}],
                "open_questions": [],
            },
        },
        "frontier": ["边界A"],
    }
    base.update(overrides)
    return base


def test_fresh_complete_state_can_exit():
    st = evaluate(make_state(), today=TODAY)
    assert st.can_exit
    assert "✅" in format_status(st)


def test_stale_evidence_blocks_exit():
    state = make_state()
    state["nodes"]["n1"]["evidence"] = [{"claim": "old", "asof": "2026-01-01"}]
    st = evaluate(state, today=TODAY)
    assert not st.can_exit
    assert st.stale_nodes and st.stale_nodes[0][2] > 90


def test_open_question_blocks_exit_deferred_does_not():
    state = make_state()
    state["nodes"]["n1"]["open_questions"] = [
        {"q": "未决", "status": "open"},
    ]
    assert not evaluate(state, today=TODAY).can_exit

    state["nodes"]["n1"]["open_questions"] = [
        {"q": "搁置", "status": "deferred", "reason": "下轮"},
    ]
    st = evaluate(state, today=TODAY)
    assert st.can_exit and st.deferred


def test_missing_triggers_is_structural_gap():
    state = make_state()
    state["nodes"]["n1"]["triggers"] = []
    st = evaluate(state, today=TODAY)
    assert not st.can_exit
    assert any("触发器" in what for _, what in st.incomplete)


def test_warning_alert_does_not_block_but_shows():
    state = make_state()
    state["nodes"]["n1"]["triggers"][0]["status"] = "warning"
    st = evaluate(state, today=TODAY)
    assert st.can_exit          # 论点信号不阻塞调研退出
    assert st.alerts and "🟡" in format_status(st)


def test_stale_trigger_check_blocks_exit():
    state = make_state()
    state["global_triggers"][0]["last_checked"] = "2026-01-01"
    st = evaluate(state, today=TODAY)
    assert not st.can_exit and st.stale_triggers


def test_repo_state_loads_and_evaluates():
    """repo 自带的 research_state.yaml 必须可解析，且结构完备（无 incomplete）。"""
    st = evaluate(load_state(), today=dt.date(2026, 6, 15))
    assert st.n_nodes >= 8
    assert not st.incomplete          # 每个节点都有证据+触发器（test_equipment 除外则失败）
    assert st.open_questions          # 当前确实有未决问题（估值验证遗留）
    assert not st.can_exit            # 所以本轮还不能退出——符合实际
