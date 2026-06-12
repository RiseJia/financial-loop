"""调研循环状态机：把"调研何时算完成"变成机器可检查的判定。

方法论（详见 docs/research_loop.md）：
  调研永远不会"完成"，只能满足本轮退出条件。退出条件三层：
    1. 完备性：范围内每个节点有 evidence + triggers（结构 lint）
    2. 新鲜度：证据/触发器核对时间在保鲜期内（默认 90 天 = 一个财报季）
    3. 未决问题：open 状态的问题清零（answered 或显式 deferred）
  三层全绿 → 本轮可退出，状态交接给监控（日报 + 触发器面板）。
  任何一层不绿 → 输出的就是下一轮任务清单。

数据源：config/research_state.yaml
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import repo_root


@dataclass
class LoopStatus:
    asof: dt.date
    cadence_days: int
    stale_nodes: list[tuple[str, str, int]] = field(default_factory=list)   # (node, label, 过期天数)
    stale_triggers: list[tuple[str, str, int]] = field(default_factory=list)  # (node, condition, 过期天数)
    alerts: list[tuple[str, str, str, str]] = field(default_factory=list)   # (node, condition, status, note)
    open_questions: list[tuple[str, str]] = field(default_factory=list)     # (node, question)
    deferred: list[tuple[str, str, str]] = field(default_factory=list)      # (node, question, reason)
    incomplete: list[tuple[str, str]] = field(default_factory=list)         # (node, 缺什么)
    frontier: list[str] = field(default_factory=list)
    n_nodes: int = 0

    @property
    def can_exit(self) -> bool:
        """本轮退出判定：无过期、无未核触发器、无 open 问题、无结构缺失。

        注意：alerts（warning/fired）不阻塞退出——它们是论点层信号，
        交给监控与人工裁决；调研退出管的是"信息是否完备新鲜"。
        """
        return not (self.stale_nodes or self.stale_triggers
                    or self.open_questions or self.incomplete)


def _parse_date(v) -> dt.date | None:
    if isinstance(v, dt.date):
        return v
    if isinstance(v, str):
        try:
            return dt.date.fromisoformat(v[:10])
        except ValueError:
            return None
    return None


def load_state(path: str | Path | None = None) -> dict:
    if path is None:
        path = repo_root() / "config" / "research_state.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def evaluate(state: dict, today: dt.date | None = None) -> LoopStatus:
    today = today or dt.date.today()
    cadence = int(state.get("meta", {}).get("cadence_days", 90))
    st = LoopStatus(asof=today, cadence_days=cadence,
                    frontier=list(state.get("frontier", [])))

    def check_triggers(owner: str, triggers: list):
        for trig in triggers or []:
            cond = trig.get("condition", "?")
            checked = _parse_date(trig.get("last_checked"))
            age = (today - checked).days if checked else 10**6
            if age > cadence:
                st.stale_triggers.append((owner, cond, age))
            status = trig.get("status", "ok")
            if status in ("warning", "fired"):
                st.alerts.append((owner, cond, status, trig.get("note", "")))

    check_triggers("global", state.get("global_triggers", []))

    for key, node in (state.get("nodes") or {}).items():
        st.n_nodes += 1
        label = node.get("label", key)
        evidence = node.get("evidence") or []
        if not evidence:
            st.incomplete.append((label, "无证据条目"))
        else:
            newest = max((d for e in evidence if (d := _parse_date(e.get("asof")))),
                         default=None)
            age = (today - newest).days if newest else 10**6
            if age > cadence:
                st.stale_nodes.append((key, label, age))
        if not (node.get("triggers") or []):
            st.incomplete.append((label, "无证伪触发器"))
        check_triggers(label, node.get("triggers", []))
        for q in node.get("open_questions") or []:
            if q.get("status") == "open":
                st.open_questions.append((label, q.get("q", "?")))
            elif q.get("status") == "deferred":
                st.deferred.append((label, q.get("q", "?"), q.get("reason", "未注明")))
    return st


def format_status(st: LoopStatus) -> str:
    lines = [
        f"# 调研循环状态 — {st.asof}（保鲜期 {st.cadence_days} 天，{st.n_nodes} 个节点）",
        "",
        f"## 本轮退出判定：{'✅ 可退出（交接给监控）' if st.can_exit else '❌ 不可退出——以下即为下一轮任务清单'}",
        "",
    ]
    if st.incomplete:
        lines += ["### 结构缺失（完备性不达标）", ""]
        lines += [f"- {label}：{what}" for label, what in st.incomplete] + [""]
    if st.stale_nodes:
        lines += ["### 证据过期（需重新核实数字）", ""]
        lines += [f"- {label}：最新证据已 {age} 天" for _, label, age in st.stale_nodes] + [""]
    if st.stale_triggers:
        lines += ["### 触发器超期未核对", ""]
        lines += [f"- [{owner}] {cond}（{age} 天未核）" for owner, cond, age in st.stale_triggers] + [""]
    if st.open_questions:
        lines += ["### 未决问题（open，阻塞退出）", ""]
        lines += [f"- [{owner}] {q}" for owner, q in st.open_questions] + [""]
    if st.alerts:
        lines += ["### ⚠️ 论点信号（不阻塞退出，需人工裁决）", ""]
        lines += [f"- {'🟡' if s == 'warning' else '🔴'} [{owner}] {cond}"
                  + (f" —— {note}" if note else "")
                  for owner, cond, s, note in st.alerts] + [""]
    if st.deferred:
        lines += ["### 已搁置（deferred，下轮复议）", ""]
        lines += [f"- [{owner}] {q}（理由：{r}）" for owner, q, r in st.deferred] + [""]
    if st.frontier:
        lines += ["### 探索边界（饱和后资源投向）", ""]
        lines += [f"- {f}" for f in st.frontier] + [""]
    lines += [
        "---",
        "退出 ≠ 结束：可退出意味着状态完备新鲜，监控接管（日报+触发器）；",
        "下一轮由三类事件触发：① 保鲜期到期（财报季） ② 触发器变 warning/fired ③ 新节点发现。",
    ]
    return "\n".join(lines)
