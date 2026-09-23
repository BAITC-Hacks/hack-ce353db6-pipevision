"""Actual progress stays truthful during retries, failures and rerenders."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
from motion import mode_activation_markup, progress_markup


def states(markup):
    return re.findall(r'<li data-state="([^"]+)"', markup)


def test_progress_does_not_invent_completed_stages():
    markup = progress_markup([{"tool": "fetch_weather", "status": "ok"}], active_tool="prepare_data")
    assert states(markup) == ["ok", "running", "pending", "pending"]
    assert "fetch_weather" not in markup and "prepare_data" not in markup


def test_failure_and_skips_cannot_be_presented_as_success():
    markup = progress_markup([
        {"tool": "fetch_weather", "status": "ok"}, {"tool": "prepare_data", "status": "skipped"},
        {"tool": "run_model", "status": "ok"}, {"tool": "llm", "status": "error"},
        {"tool": "write_report", "status": "ok"},
    ], running=False)
    assert states(markup) == ["ok", "skip", "ok", "error"]
    assert "Нужна проверка" in markup and "Расчёт готов" not in markup


def test_latest_retry_result_wins_and_stopped_work_cannot_pulse():
    steps = [{"tool": "fetch_weather", "status": "error"}, {"tool": "fetch_weather", "status": "ok"},
             {"tool": "prepare_data", "status": "running"}]
    assert states(progress_markup(steps, running=False)) == ["ok", "pending", "pending", "pending"]


def test_unknown_status_remains_pending():
    assert states(progress_markup([{"tool": "run_model", "status": "unexpected"}], running=False)) == ["pending"] * 4


def test_completion_and_mode_ack_are_accessible():
    steps = [{"tool": t, "status": "ok"} for t in ("fetch_weather", "prepare_data", "run_model", "write_report")]
    markup = progress_markup(steps, running=False)
    assert states(markup) == ["ok"] * 4
    assert 'role="status">Расчёт готов' in markup
    assert "prefers-reduced-motion:reduce" in markup
    activation = mode_activation_markup()
    assert '.9s ease-out 1' in activation and "infinite" not in activation
    assert 'role="status"' in activation and "prefers-reduced-motion:reduce" in activation


def test_browser_card_uses_real_states_and_acknowledges_each_action_once():
    """Exercise the embedded JS without a browser or extra frontend dependencies."""
    import json
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is optional; card state integration check requires Node")
    template = (Path(__file__).resolve().parents[1] / "ui/assets/agent_card.html").read_text()
    script = re.search(r"<script>([\s\S]*?)</script>", template).group(1)
    steps = [{"tool": t, "status": "ok"} for t in ("fetch_weather", "prepare_data", "run_model", "write_report")]
    cases = [
        {"key": "run-1", "steps": steps, "animate": False},
        {"key": "run-2", "steps": steps, "animate": True},
        {"key": "run-2", "steps": steps, "animate": True},
        {"key": "run-3", "steps": steps, "animate": True, "reduce": True},
        {"key": "run-4", "steps": steps + [{"tool": "llm", "status": "error"}], "decision": {"code": "accept"}},
        {"key": "run-5", "steps": steps[:2], "animate": True},
        {"key": "run-6", "steps": steps + [{"tool": "verify_narrative", "status": "skipped"}]},
    ]
    harness = r"""
const vm=require('node:vm'), input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const seen={};
function element(){return {dataset:{},children:[],classes:[],classList:{add(c){this.parent.classes.push(c)}},appendChild(c){this.children.push(c)},setAttribute(){}};}
const results=input.cases.map(data=>{
  const els={strip:element(),status:element(),stages:element()};
  Object.values(els).forEach(e=>e.classList.parent=e);
  const document={documentElement:element(),getElementById:id=>els[id],createElement:()=>element()};
  const window={matchMedia:()=>({matches:!!data.reduce}),sessionStorage:{getItem:k=>seen[k],setItem:(k,v)=>seen[k]=v}};
  vm.runInNewContext(input.script.replace('/*__WA_CARD_DATA__*/null',JSON.stringify(data)),{document,window});
  return {states:els.stages.children.map(e=>e.dataset.state),status:els.status.textContent,animate:els.strip.classes.includes('ack')};
});
process.stdout.write(JSON.stringify(results));
"""
    result = subprocess.run([node, "-e", harness], input=json.dumps({"script": script, "cases": cases}),
                            text=True, capture_output=True, check=True)
    out = json.loads(result.stdout)
    assert [c["animate"] for c in out] == [False, True, False, False, False, False, False]
    assert out[4]["states"] == ["ok", "ok", "ok", "error"]
    assert out[4]["status"] == "Нужна проверка"
    assert out[5]["states"] == ["ok", "ok", "pending", "pending"]
    assert out[6]["states"] == ["ok", "ok", "ok", "skip"]
    assert out[6]["status"] == "Есть ограничения"
