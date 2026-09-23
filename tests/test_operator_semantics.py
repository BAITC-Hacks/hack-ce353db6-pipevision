"""Operator decisions must refer to the visible period, in physical units, without implied guarantees."""
from __future__ import annotations

import json
import hashlib
import functools
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
core = pytest.importorskip("core")


@pytest.fixture
def result(monkeypatch):
    history = {"by_month": {s: {2: .4} for s in ("t1", "t2", "farm")}, "range": ["2024-02-01", "2026-01-31"]}
    monkeypatch.setattr(core.tools, "load_history", lambda: history)
    ts = pd.date_range("2026-02-07", periods=48, freq="h")
    p = np.repeat([.2, .9], 24)
    frames = []
    for series in ("t1", "t2", "farm"):
        frames.append(pd.DataFrame({"turbine": series, "issue_date": "2026-02-06",
                                    "target_time_local": ts.strftime("%Y-%m-%dT%H:%M:%S+05:00"),
                                    "target_time_utc": (ts - pd.Timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    "lead_hours": np.arange(1, 49), "lead_day": np.repeat([1, 2], 24),
                                    "p50": p, "p10": p - .1, "p90": p + .05,
                                    "wind_speed_100m": 6., "wind_direction_100m": 90.}))
    fc = pd.concat(frames, ignore_index=True)
    a = core.tools.analyze_forecast(fc, history=history)
    return core.ForecastResult(core.ForecastParams(), core.nurly_site(), fc, a,
                               {"decision": "flag", "narrative": "Полный выпуск: 132 МВт·ч за 48 ч", "reasoning": "рампа 08.02"},
                               [], {}, "")


def test_selected_horizon_is_shared_by_summary_card_alerts_and_chat(result):
    a = core.horizon_analysis(result, 24)
    assert a["metrics"]["farm"]["mean_p50"] == .2
    assert a["checks"]["complete"]
    assert not any(f["code"] in {"ramp", "incomplete"} for f in a["flags"])
    summary = core.operator_summary(result, 24)
    assert summary["energy_mwh"] == 24.0
    assert summary["mean_mw"] == 1.0
    card = core.forecast_card(result, 24)
    assert "24 ч" in card["subject"] and "24.0 МВт·ч" in card["message"]
    assert "132" not in card["message"] and "08.02" not in card["reasoning"]
    assert card["decision"]["code"] == "accept" and card["decision"]["word"] != "ACCEPT"
    assert not any(r.get("code") == "ramp" for r in core.alert_rows(result, result.view(24)))
    answer = core.rules_answer("Почему такое решение?", result, None, 24)
    assert "24 ч" in answer and "08.02" not in answer
    assert any(f["code"] == "ramp" for f in core.horizon_analysis(result, 48)["flags"])


def test_missing_history_does_not_restore_full_horizon(result, monkeypatch):
    def unavailable():
        raise OSError("SCADA missing")
    monkeypatch.setattr(core.tools, "load_history", unavailable)
    a = core.horizon_analysis(result, 24)
    assert a["metrics"]["farm"]["energy_48h_mwh"] == 24.0
    assert a["metrics"]["farm"]["climatology_mean"] is None


@pytest.mark.parametrize("level", ["error", "critical"])
def test_core_error_cannot_produce_a_ready_summary_or_accept_card(result, level):
    result.core_log.append((level, "backend error"))
    summary = core.operator_summary(result, 24)
    assert summary["status"] == "error" and "Расчёт сообщил об ошибке" in summary["message"]
    assert core.forecast_card(result, 24)["decision"]["code"] == "recalculate"


def test_wide_forecast_explains_warning_even_though_core_flag_is_info(result):
    result.forecast.loc[result.forecast.lead_hours <= 24, ["p10", "p90"]] = [0., 1.]
    summary = core.operator_summary(result, 24)
    assert summary["status"] == "warning" and "Разброс прогноза велик" in summary["message"]


def test_analysis_failure_is_explicit_not_full_period_fallback(result, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ValueError("invalid forecast")
    monkeypatch.setattr(core.tools, "analyze_forecast", unavailable)
    a = core.horizon_analysis(result, 24)
    assert not a["metrics"] and a["flags"][0]["code"] == "analysis_unavailable"
    assert core.operator_summary(result, 24)["status"] == "error"
    assert core.alert_rows(result, result.view(24))[0]["level"] == "error"
    assert "None" not in core.rules_answer("Сколько энергии?", result, None, 24)


def test_llm_context_contains_scoped_physical_units_without_full_narrative(result):
    ctx, _, fc = core.qa_context(result, None, 24)
    forecast = ctx["прогноз"]
    assert forecast["горизонт, ч"] == 24
    assert forecast["показатели"]["выработка выбранного периода, МВт·ч"] == 24.0
    assert forecast["показатели"]["выработка по календарным дням"] == [
        {"дата": "2026-02-07", "часов в выбранном периоде": 24, "МВт·ч": 24.0}]
    text = json.dumps(ctx, ensure_ascii=False)
    assert "energy_day2" not in text and "132" not in text
    assert "не являются калиброванным интервалом энергии" in text
    assert fc["lead_hours"].max() == 24


def test_ai_switch_off_never_reads_key_settings(result, monkeypatch):
    def forbidden():
        raise AssertionError("LLM should not be consulted")
    monkeypatch.setattr(core.llm, "llm_settings", forbidden)
    out = core.ask_agent("Сколько энергии?", result, horizon=24, use_llm=False)
    assert not out["llm_used"] and "24.0 МВт·ч" in out["answer"]
    assert "ч.н." not in out["answer"] and out["fact_check"]["ok"]


@pytest.mark.parametrize("question", ["Насколько точный прогноз?", "Насколько точный прогноз на 24 ч?", "Какой интервал за 48 ч?"])
def test_quality_intent_is_not_mistaken_for_energy(result, question):
    answer = core.rules_answer(question, result, None, 24)
    assert "не оценка точности" in answer
    assert "Ожидаемая выработка" not in answer


@pytest.mark.parametrize(("question", "expected"), [("Какие рампы на 24 ч?", "перепад"),
                                                   ("Когда ТО за 48 ч?", "ТО"),
                                                   ("Когда пик выработки?", "Пик мощности")])
def test_specific_question_takes_precedence_over_generic_energy_words(result, question, expected):
    answer = core.rules_answer(question, result, None, 24)
    assert expected in answer and "Ожидаемая выработка" not in answer


@pytest.mark.parametrize("response", ["Ожидается 987654 МВт·ч", "energy_day2 = 24.0 МВт·ч"])
def test_unverified_or_technical_llm_answer_is_replaced(result, monkeypatch, response):
    monkeypatch.setattr(core.llm, "llm_settings", lambda: {"model": "fake"})
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))]))))
    monkeypatch.setattr(core.llm, "make_client", lambda: client)
    out = core.ask_agent("Сколько энергии?", result, horizon=24)
    assert not out["llm_used"] and out["fact_check"]["ok"]
    assert "energy_day2" not in out["answer"] and "987654" not in out["answer"]


def test_direction_proxy_does_not_claim_no_wake_losses(result, monkeypatch):
    monkeypatch.setattr(core, "park_layout", lambda: {"present": True})
    monkeypatch.setattr(core, "upwind_count", lambda wd, layout: 0)
    direction = next(row for row in core.why_lines(result, result.view(24)) if row[0] == "Направление")
    assert "не оценка потерь" in direction[2]
    assert "потерь в следе не ожидается" not in direction[2]


def test_missing_hours_do_not_form_a_one_hour_ramp_or_maintenance_window():
    n = core.CALM_MIN_HOURS
    times = pd.date_range("2026-02-07", periods=n, freq="2h")
    frame = pd.DataFrame({"lead_hours": range(1, n + 1), "time_local": times, "p50": np.zeros(n)})
    assert not core.calm_windows(frame)
    frame.loc[1, "p50"] = 1.0
    assert not core.ramp_windows(frame)


def test_steady_card_does_not_read_video(result, monkeypatch):
    def forbidden():
        raise AssertionError("Do not send a decorative video with every forecast")
    monkeypatch.setattr(core, "intro_video_uri", forbidden)
    core.agent_card_html(core.forecast_card(result, 24))


@pytest.fixture
def validation_files(tmp_path):
    paths = ["models/power_model.joblib", "src/wind_agent/data.py", "src/wind_agent/features.py", "src/wind_agent/model.py"]
    for rel in paths:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rel)
    metrics = {"test_range": ["2025-12-01", "2026-01-31"], "train_range": ["2024-02-16", "2025-11-30"],
               "calibration": {"range": ["2025-10-01", "2025-11-30"]},
               "overall": {"model": {"mae": .17, "rmse": .24, "bias": .03, "n": 100}, "coverage_p10_p90_pct": 75.},
               "by": {"t1_lead1": {"model": {"mae": .16}}}}
    metrics_path = tmp_path / "outputs/backtest_metrics.json"
    metrics_path.parent.mkdir()
    metrics_path.write_text(json.dumps(metrics))
    sha = lambda rel: hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest()
    manifest = {"schema_version": 1, "created_at_utc": "2026-02-01T12:00:00Z", "model_sha256": sha(paths[0]),
                "metrics_file": "outputs/backtest_metrics.json", "metrics_sha256": sha("outputs/backtest_metrics.json"),
                "source_sha256": {rel: sha(rel) for rel in paths[1:]}, "protocol": "historical holdout with final refit",
                "train_end": "2025-11-30", "test_period": {"start": "2025-12-01", "end": "2026-01-31"},
                "calibration_period": {"start": "2025-10-01", "end": "2025-11-30"},
                "artifact_train_end": "2026-01-31",
                "artifact_calibration_period": {"start": "2025-12-01", "end": "2026-01-31"},
                "limitations": ["Historical validation does not measure the selected forecast."]}
    (tmp_path / "models/validation_manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_historical_validation_reports_data_age_not_manifest_age(validation_files):
    out = core.historical_validation(validation_files, today=date(2026, 2, 10))
    assert out["status"] == "verified" and out["test_age_days"] == 10
    assert out["metrics"]["overall"]["model"]["mae"] == .17
    assert "не измерение точности выбранного выпуска" in out["message"]


@pytest.mark.parametrize("changed", ["models/power_model.joblib", "src/wind_agent/data.py", "src/wind_agent/features.py",
                                    "src/wind_agent/model.py", "outputs/backtest_metrics.json"])
def test_validation_is_withheld_if_model_code_or_metrics_changed(validation_files, changed):
    with (validation_files / changed).open("a") as f:
        f.write(" ")
    out = core.historical_validation(validation_files, today=date(2026, 2, 10))
    assert out["status"] == "mismatch" and out["metrics"] is None


def test_legacy_metrics_without_model_manifest_are_not_current_validation(validation_files):
    (validation_files / "models/validation_manifest.json").unlink()
    out = core.historical_validation(validation_files)
    assert out["status"] == "unavailable" and out["metrics"] is None


def test_validation_with_overlapping_train_test_is_rejected(validation_files):
    path = validation_files / "models/validation_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["train_end"] = "2025-12-01"
    path.write_text(json.dumps(manifest))
    out = core.historical_validation(validation_files, today=date(2026, 2, 10))
    assert out["status"] == "invalid" and out["metrics"] is None


@pytest.mark.parametrize("payload", [[], None, {"schema_version": 1, "source_sha256": []}])
def test_malformed_validation_metadata_fails_closed(validation_files, payload):
    (validation_files / "models/validation_manifest.json").write_text(json.dumps(payload))
    out = core.historical_validation(validation_files)
    assert out["status"] == "invalid" and out["metrics"] is None


def test_validation_does_not_attach_new_model_metrics_to_old_forecast(validation_files):
    out = core.historical_validation(validation_files, today=date(2026, 2, 10), forecast_model_sha256="0" * 64)
    assert out["status"] == "mismatch" and out["metrics"] is None
    assert "прогноз предыдущей версии" in out["message"]


def test_replacing_artifact_invalidates_process_model_cache(tmp_path, monkeypatch):
    path = tmp_path / "power_model.joblib"
    path.write_bytes(b"first model")
    monkeypatch.setattr(core.config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(core, "_LOADED_MODEL_SHA", None)

    @functools.lru_cache(maxsize=1)
    def load():
        return SimpleNamespace(meta={"train_end": "2026-01-31"}, data=path.read_bytes())

    monkeypatch.setattr(core.tools, "load_model", load)
    monkeypatch.setattr(core.tools, "load_history", lambda: {})
    first = core.load_core()
    assert core.load_core()["model"] is first["model"]
    path.write_bytes(b"second model")
    second = core.load_core()
    assert second["model"].data == b"second model"
    assert second["model_sha256"] != first["model_sha256"]
