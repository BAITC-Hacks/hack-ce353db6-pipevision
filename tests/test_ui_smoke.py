"""Дымовые тесты панели оператора: мост ui/core.py (без сети) и запуск страницы через streamlit AppTest."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

UI = Path(__file__).resolve().parents[1] / "ui"
sys.path.insert(0, str(UI))
core = pytest.importorskip("core")


def test_kz_outline_and_point_check():
    assert core.kz_outline(), "контур Казахстана (atlas.kz_polygon или data/atlas/kaz.geo.json)"
    assert core.in_kz(43.6, 78.3)            # ВЭС «Нурлы»
    assert core.in_kz(51.17, 71.43)          # Астана
    assert not core.in_kz(55.7, 49.0)        # Казань — вне Казахстана


def test_fallback_centers_inside_kz():
    c = core.fallback_centers()
    assert len(c) > 500 and {"lat", "lon"} <= set(c.columns)
    assert c["lat"].between(*core.KZ_LAT).all() and c["lon"].between(*core.KZ_LON).all()


def test_study_params_key_and_label():
    p = core.StudyParams(lat=47.123456, lon=51.987654, n_turbines=20, rated_mw=4.5)
    assert p.key() == (47.123, 51.988, 20, 4.5)
    assert "47.12" in p.label and "51.99" in p.label
    assert core.StudyParams(47.0, 52.0, name="Атырау-1").label == "Атырау-1"


def test_run_assessment_without_backend_modules(monkeypatch):
    """Без atlas/assess/vizdata шаги помечаются «пропущен», прогноз строится, исключения не выходят наружу."""
    monkeypatch.setattr(core, "_optional", lambda name: None)
    sentinel = object()
    monkeypatch.setattr(core, "run_forecast", lambda fp, wx=None, progress=None: sentinel)
    seen = []
    out = core.run_assessment(core.StudyParams(47.0, 52.0), wx=object(), progress=lambda i, lab, st: seen.append((i, st)))
    assert out["forecast_result"] is sentinel
    assert out["assessment"] is None and out["viz_html"] is None and out["report"] is None
    assert set(out["skipped"]) == {core.STUDY_STEPS[i] for i in (0, 1, 3, 4)}
    assert (2, "done") in seen and not out["errors"]


def test_run_assessment_step_error_is_short(monkeypatch):
    monkeypatch.setattr(core, "_optional", lambda name: None)

    def boom(fp, wx=None, progress=None):
        raise RuntimeError("нет сети")
    monkeypatch.setattr(core, "run_forecast", boom)
    out = core.run_assessment(core.StudyParams(47.0, 52.0), wx=object())
    assert out["errors"][core.STUDY_STEPS[2]] == "RuntimeError: нет сети"


def test_app_renders_offline(monkeypatch):
    """Первый экран и два рабочих пространства доступны без API-запросов."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    client_type = core.WeatherClient
    monkeypatch.setattr(core, "WeatherClient", lambda **kw: client_type(offline=True))
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.title and "Нурлы" in at.title[0].value
    assert "result" in at.session_state and isinstance(at.session_state["result"].forecast, pd.DataFrame)
    assert at.chat_input
    assert len(at.metric) == 3
    assert {e.label for e in at.expander} >= {"Спросить агента", "Почему такой прогноз и что изменилось", "Данные и проверка качества"}
    # одна страница: карта Казахстана с формой площадки рядом с прогнозом «Нурлы», точка по умолчанию — «Нурлы»
    labels = [b.label for b in at.button]
    assert "Исследовать площадку" in labels and "Вернуться к «Нурлы»" in labels and "Рассчитать" in labels
    assert abs(float(at.session_state["cand_lat"]) - 43.64) < 0.05 and abs(float(at.session_state["cand_lon"]) - 78.54) < 0.05
    assert not at.sidebar.children if hasattr(at.sidebar, "children") else True   # параметры выпуска — на странице


def test_operator_horizon_draft_and_recalculation(monkeypatch):
    """Смена горизонта фильтрует результат; дата остаётся черновиком до явного расчёта."""
    from datetime import date
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("OPENAI_API_KEY", "")
    client_type = core.WeatherClient
    monkeypatch.setattr(core, "WeatherClient", lambda **kw: client_type(offline=True))
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=120).run()
    assert not at.exception
    result = at.session_state["result"]
    at.radio(key="horizon").set_value(24).run()
    assert not at.exception
    assert at.session_state["result"] is result
    assert at.metric[0].label == "Выработка за 24 ч"
    expected = core.operator_summary(result, 24)["message"]
    assert any(expected in m.value for m in at.markdown)
    at.session_state["issue_date"] = date(2026, 2, 6)
    at.run()
    assert any("Параметры изменены" in w.value for w in at.warning)
    assert at.session_state["result"] is result
    next(b for b in at.button if b.label == "Рассчитать").click().run()
    assert not at.exception
    assert at.session_state["result"].issue_label == "2026-02-06"
    assert not any("Параметры изменены" in w.value for w in at.warning)
    at.radio(key="detail_section").set_value("Факт и точность").run()
    assert not at.exception
    validation = core.historical_validation(forecast_model_sha256=at.session_state["result"].meta["model_sha256"])
    assert validation["status"] == "verified"
    metric = next(m for m in at.metric if m.label == "Средняя ошибка, % номинала")
    assert metric.value == f"{100 * validation['metrics']['overall']['model']['mae']:.1f}"
    assert any("не измерение точности выбранного выпуска" in c.value for c in at.caption)
    at.chat_input[0].set_value("Насколько точный прогноз?").run()
    assert not at.exception
    assert any("не оценка точности" in m.value for m in at.markdown)


@pytest.fixture(scope="module")
def nurly_result():
    """Выпуск «Нурлы» 10.02.2026 из кэша Open-Meteo, без LLM."""
    return core.run_forecast(core.ForecastParams())


def test_ask_agent_rules_without_key(monkeypatch, nurly_result):
    """Без ключа OpenAI агент отвечает по правилам: энергия за 48 ч — числом из анализа, числа проходят сверку."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = core.ask_agent("Сколько энергии выработает парк за 48 ч?", nurly_result)
    m = nurly_result.analysis["metrics"]["farm"]
    assert out["llm_used"] is False
    assert str(m["energy_48h_mwh"]) in out["answer"] and "МВт·ч" in out["answer"]
    assert out["fact_check"]["checked"] > 0 and not out["fact_check"]["unverified"]
    other = core.ask_agent("Расскажи анекдот", nurly_result)
    assert "ключ OpenAI" in other["answer"]


def test_agent_card_html_forecast_and_study(nurly_result):
    """HTML карточки агента строится для выпуска и для исследования площадки; данные подставлены вместо метки."""
    htm = core.agent_card_html(core.forecast_card(nurly_result), animate=True, intro=True, nonce="t")
    assert "Агент WindAgent" in htm and core.CARD_DATA_MARK not in htm and "fetch_weather" in htm
    study = {"params": core.StudyParams(47.0, 52.0, name="Атырау-1"), "assessment": None, "forecast_result": nurly_result,
             "viz_html": None, "report": {"markdown": "# Оценка\n\n## Резюме\n\nПлощадка **перспективная**.\n\n## Площадка\n\nx",
                                          "llm_used": False, "fact_check": {"checked": 3, "unverified": []}},
             "errors": {core.STUDY_STEPS[1]: "RuntimeError: нет сети"}, "skipped": [], "timings": {}}
    card = core.study_card(study)
    assert [s["status"] for s in card["steps"]][:2] == ["ok", "error"] and card["message"] == "Площадка перспективная."
    assert core.CARD_DATA_MARK not in core.agent_card_html(card, animate=False)
