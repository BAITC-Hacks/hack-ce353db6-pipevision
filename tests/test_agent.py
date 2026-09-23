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


# ---------------------------------------------------------------- проверка фактов в нарративе
TEST_DATES = [d.strftime("%Y-%m-%d") for d in pd.date_range(config.TEST_ISSUE_START, config.TEST_ISSUE_END)]
FAKE = " Кроме того, выработка составит 987.6 часа номинала при ветре 77.7 м/с."


@pytest.fixture(scope="module")
def analyses(wx):
    """Анализ и прогноз по всем датам тестового периода (ревизия — к предыдущему выпуску), без записи файлов."""
    out, prev = {}, None
    for d in TEST_DATES:
        prepared = tools.prepare_data(tools.fetch_weather(wx, d))
        fc = tools.run_model(prepared)
        out[d] = (tools.analyze_forecast(fc, prev, notes=prepared["notes"], info=prepared["info"]), fc)
        prev = fc
    return out


def test_template_narrative_always_passes_fact_check(analyses):
    for d, (a, fc) in analyses.items():
        res = tools.verify_narrative(tools.template_narrative(a), a, fc)
        assert res["ok"], (d, res["unverified"])
        assert res["checked"] >= 10


def test_fabricated_numbers_are_rejected(analyses):
    a, fc = analyses[ISSUE]
    res = tools.verify_narrative(tools.template_narrative(a) + FAKE, a, fc)
    assert not res["ok"]
    assert res["unverified"] == ["987.6", "77.7"]


def test_extract_numbers_formats():
    nums, skipped = tools.extract_numbers(
        "Смещение −0,03; (−9.7 %) выше нормы; 5.5–28.5; D+1, P50, t1; 2026-02-11 06:00; 11 февраля 2026 года")
    assert [(v, p) for _, v, p in nums] == [(-0.03, False), (-9.7, True), (5.5, False), (28.5, False)]
    assert skipped >= 5


@pytest.mark.parametrize("extra, action", [(" Выработка составит 987.6 часа номинала.", "annotated"), (FAKE, "replaced")])
def test_llm_narrative_fact_check_in_report(wx, tmp_outputs, extra, action):
    """1 неподтверждённое число — пометка в отчёте; 2 и более — нарратив LLM заменяется шаблоном."""
    s = orchestrator.AgentSession(wx, issue_date=ISSUE)
    for t in orchestrator.PIPELINE:
        s.call(t)
    template = tools.template_narrative(s.state["analysis"])
    s.call("write_report", by_llm=True, decision="accept", reasoning="Проверки пройдены, флагов уровня warning нет.",
           narrative_ru=template + extra)
    report = (tmp_outputs / "reports" / f"{ISSUE}.md").read_text(encoding="utf-8")
    step = [t for t in s.trace if t["tool"] == "verify_narrative"]
    assert len(step) == 1 and step[0]["status"] == "warning"
    assert s.final["fact_check"]["action"] == action
    if action == "annotated":
        assert "⚠ Не подтверждено анализом: 987.6" in report
        assert "987.6" in s.final["narrative"]
    else:
        assert "Комментарий LLM отклонён проверкой фактов" in report
        assert s.final["narrative"] == template
