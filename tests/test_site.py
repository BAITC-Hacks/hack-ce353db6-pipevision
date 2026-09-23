"""Площадка как объект, час выпуска, сверка с фактом и файл сдачи (всё офлайн)."""
import numpy as np
import pandas as pd
import pytest

from wind_agent import config, evaluate
from wind_agent.weather import WeatherClient


def test_custom_site_and_nurly():
    s = config.custom_site(51.18, 71.45, n_turbines=10, rated_mw=2.5, name="Тест")
    assert s.transfer and s.series == ["u1", "farm"] and s.n_turbines == 10 and s.key.startswith("c51p180")
    assert config.NURLY.has_history and config.NURLY.series == ["t1", "t2", "farm"]
    with pytest.raises(ValueError):
        config.custom_site(10.0, 10.0)   # вне Казахстана


def test_archived_forecast_issue_hour_offline():
    wx = WeatherClient(offline=True)
    f = wx.archived_forecast("t1", "2026-02-10", issue_hour=12)
    assert len(f) == config.HORIZON_HOURS
    assert str(f["target_local"].iloc[0]) == "2026-02-10 13:00:00"      # первый час после выпуска в 12:00
    assert f["lead_hours"].tolist() == list(range(1, 49))
    assert f["lead_day"].tolist() == [1] * 24 + [2] * 24
    # протокол ТЗ (23:00) остаётся прежним
    g = wx.archived_forecast("t1", "2026-02-10")
    assert str(g["target_local"].iloc[0]) == "2026-02-11 00:00:00"
    assert g["issue_time_utc"].iloc[0] == pd.Timestamp("2026-02-10 18:00", tz="UTC")


def test_evaluate_metrics_synthetic():
    t = pd.date_range("2026-02-01", periods=48, freq="h", tz="UTC")
    rows = []
    for turb in ("t1", "t2", "farm"):
        for lead in (1, 2):
            rows.append(pd.DataFrame({"turbine": turb, "lead_day": lead, "t_utc": t,
                                      "p10": 0.2, "p50": 0.5, "p90": 0.8}))
    fc = pd.concat(rows, ignore_index=True)
    actual = {"t1": pd.Series(0.6, index=t), "t2": pd.Series(0.4, index=t)}
    res = evaluate.evaluate(fc, actual)
    by = {(r["turbine"], r["lead_day"]): r for r in res["by"]}
    assert by[("t1", 1)]["mae"] == pytest.approx(0.1) and by[("t2", 2)]["bias"] == pytest.approx(0.1)
    assert by[("farm", 1)]["mae"] == pytest.approx(0.0)                  # факт парка = среднее турбин = 0.5
    assert res["overall"]["mae"] == pytest.approx(0.1) and res["overall"]["coverage_p10_p90_pct"] == 100.0
    assert "MAE" in evaluate.format_report(res)


def test_submission_file(tmp_path):
    path = evaluate.build_submission(out_path=tmp_path / "sub.csv")
    sub = pd.read_csv(path)
    assert len(sub) == 28 * 48 * 3 and set(sub["horizon_h"]) == {24, 48}
    assert set(sub["turbine"].astype(str)) == {"1", "2", "farm"}
    assert sub["Статистическое время"].iloc[0].startswith("2026-02-01 00:00")
    assert ((sub["p10"] <= sub["p50"]) & (sub["p50"] <= sub["p90"])).all()
