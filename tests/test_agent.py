"""Сквозной офлайн-прогон агента на одной дате выпуска. Все записи — во временную папку, outputs/ репозитория не трогаем."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from wind_agent import config
from wind_agent.agent import llm, orchestrator, tools

ISSUE = "2026-02-10"


@pytest.fixture()
def tmp_outputs(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "OUTPUTS_DIR", out)
    monkeypatch.setattr(orchestrator, "FORECASTS_DIR", out / "forecasts")
    monkeypatch.setattr(orchestrator, "REPORTS_DIR", out / "reports")
    monkeypatch.setattr(orchestrator, "LIVE_DIR", out / "live")
    monkeypatch.setattr(orchestrator, "AGENT_LOG", out / "agent_log.jsonl")
    monkeypatch.setattr(orchestrator, "REPLAY_SUMMARY", out / "replay_summary.csv")
    monkeypatch.setattr(orchestrator, "LATEST_BY_TARGET", out / "forecasts" / "latest_by_target.csv")
    return out


def _check_outputs(out, summary):
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["status"] in ("ok", "warning")
    assert row["decision"] in ("accept", "recalculate", "flag")
    assert row["api_calls"] == 0                      # офлайн: только кэш Open-Meteo
    fc = pd.read_csv(out / "forecasts" / f"{ISSUE}.csv")
    assert list(fc.columns) == tools.FORECAST_COLUMNS
    assert len(fc) == 144 and set(fc["turbine"]) == {"t1", "t2", "farm"}
    q = fc[["p10", "p50", "p90"]]
    assert ((q >= 0) & (q <= 1)).all().all()
    assert ((fc["p10"] <= fc["p50"]) & (fc["p50"] <= fc["p90"])).all()
    assert fc["lead_hours"].min() == 1 and fc["lead_hours"].max() == 48
    assert (out / "reports" / f"{ISSUE}.md").read_text(encoding="utf-8").startswith("# ")
    log = [json.loads(line) for line in (out / "agent_log.jsonl").read_text(encoding="utf-8").splitlines()]
    tools_called = [e["tool"] for e in log if e["tool"] != "llm"]
    assert tools_called[:4] == ["fetch_weather", "prepare_data", "run_model", "analyze_forecast"]
    assert tools_called[-1] == "write_report"
    return fc, log


def test_replay_one_issue_no_llm(wx, tmp_outputs):
    summary = orchestrator.run_replay(wx, ISSUE, ISSUE, use_llm=False)
    _, log = _check_outputs(tmp_outputs, summary)
    assert not bool(summary.iloc[0]["llm_used"])
    assert not any(e["llm_used"] for e in log)


def test_llm_disabled_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm.llm_settings() is None
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert llm.llm_settings()["model"] == "gpt-5-mini"


def test_llm_failure_falls_back_to_rules(wx, tmp_outputs, monkeypatch):
    """Ключ есть, но API недоступен (закрытый локальный порт) → агент доводит цикл по правилам, прогноз тот же."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-invalid")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    summary = orchestrator.run_replay(wx, ISSUE, ISSUE, use_llm=True)
    fc_llm, log = _check_outputs(tmp_outputs, summary)
    assert any(e["tool"] == "llm" and e["status"] == "error" for e in log)
    # числа прогноза не зависят от того, кто вёл цикл: сравниваем с детерминированным прогоном
    orchestrator.run_replay(wx, ISSUE, ISSUE, use_llm=False)
    fc_rules = pd.read_csv(tmp_outputs / "forecasts" / f"{ISSUE}.csv")
    pd.testing.assert_frame_equal(fc_llm[["p10", "p50", "p90"]], fc_rules[["p10", "p50", "p90"]])
