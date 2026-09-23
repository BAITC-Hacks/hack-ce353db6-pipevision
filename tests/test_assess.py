"""Оценка площадки: Weibull, КИУМ, роза ветров и отчёт для руководства на синтетическом ряде (без сети)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_agent import assess, config
from wind_agent.weather import WeatherClient


def _synthetic(seed: int, scale: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h", tz="UTC")
    ws = scale * rng.weibull(2.0, len(idx))
    return pd.DataFrame({"wind_speed_100m": ws, "wind_direction_100m": rng.choice([90.0, 270.0], len(idx), p=[0.7, 0.3]),
                         "temperature_2m": 10.0, "surface_pressure": 950.0}, index=idx)


@pytest.fixture()
def assessment(monkeypatch, tmp_path):
    curve = assess.parametric_curve()
    monkeypatch.setattr(assess, "power_curve", lambda wx=None: {"version": 2, "turbine": curve, "nurly_calibrated": curve})
    nurly = config.TURBINES["t1"]

    def fake_archive(self, lat, lon, start, end, variables=None, cache_key=None):
        return _synthetic(1, 6.0) if (round(lat, 3), round(lon, 3)) == (round(nurly[0], 3), round(nurly[1], 3)) else _synthetic(2, 9.0)

    monkeypatch.setattr(WeatherClient, "archive_hourly", fake_archive)
    wx = WeatherClient(cache_dir=tmp_path, offline=True)
    site = config.custom_site(50.0, 70.0, n_turbines=10, rated_mw=3.0, name="Тестовая площадка")
    return assess.assess_site(site, wx)


def test_weibull_moments_recovers_parameters():
    rng = np.random.default_rng(0)
    w = assess.weibull_moments(8.0 * rng.weibull(2.0, 200_000))
    assert abs(w["k"] - 2.0) < 0.05 and abs(w["c"] - 8.0) < 0.1


def test_parametric_curve_shape():
    c = assess.parametric_curve()
    p = assess.apply_curve(c, np.array([2.0, 3.0, 12.0, 20.0, 26.0]))
    assert p[0] == 0 and p[1] == 0 and p[2] == pytest.approx(1.0) and p[3] == pytest.approx(1.0) and p[4] == 0


def test_assess_site_synthetic(assessment):
    a = assessment
    assert a["period"]["hours"] == 8760
    assert 7.5 < a["ws100"]["mean"] < 8.5                     # 9 · Γ(1.5) ≈ 7.98
    assert abs(a["weibull"]["k"] - 2.0) < 0.1
    assert 0 < a["cf"] < 1
    assert a["aep_gwh"] == pytest.approx(10 * 3.0 * a["cf"] * 8760 / 1000, rel=0.01)
    assert a["aep_per_turbine_gwh"] == pytest.approx(3.0 * a["cf"] * 8760 / 1000, rel=0.01)
    assert len(a["monthly"]) == 12 and len(a["diurnal"]) == 24 and len(a["rose"]) == 16
    assert a["prevailing_sector"] == "В"
    assert sum(r["share"] for r in a["rose"]) == pytest.approx(1.0, abs=0.01)
    assert a["density_ratio"] == pytest.approx(950 * 100 / (287.05 * 283.15) / 1.225, abs=0.002)
    assert a["benchmark"]["ratio_to_nurly"] == pytest.approx(1.5, abs=0.05)
    assert a["elevation_m"] is None                           # офлайн, кэша нет
    assert a["atlas"]["resource_class"] in ("низкий", "умеренный", "хороший", "отличный")
    import json
    json.dumps(a)                                             # JSON-сериализуемо


def test_management_report_template(assessment, tmp_path):
    a = assessment
    rep = assess.management_report(a, use_llm=False)
    md = rep["markdown"]
    assert rep["llm_used"] is False
    for sec in ("## Резюме", "## Площадка", "## Ветровой ресурс", "## Ожидаемая выработка и сравнение с «Нурлы»",
                "## Сезонность и режим", "## Риски и ограничения", "## Рекомендация"):
        assert sec in md
    assert f"{a['ws100']['mean']:.2f} м/с" in md
    assert f"{a['aep_gwh']:.2f} ГВт·ч" in md
    assert f"{a['cf'] * 100:.1f} %" in md
    assert rep["fact_check"]["ok"], rep["fact_check"]         # шаблон проходит собственную проверку чисел
    path = assess.write_assessment(a, tmp_path)
    assert path.exists() and path.name == "assessment.json"


def test_fact_check_rejects_invented_number(assessment):
    from wind_agent.agent.tools import verify_narrative
    facts = assess.allowed_facts(assessment)
    bad = verify_narrative("Окупаемость проекта 6.7 лет при тарифе 34.61 тенге.", facts)
    assert not bad["ok"]
