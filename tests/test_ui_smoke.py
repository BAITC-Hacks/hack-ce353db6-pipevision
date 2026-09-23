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


def test_app_renders_offline():
    """Главная страница открывается: прогноз «Нурлы» из кэша, карта, форма площадки, вкладки — без исключений."""
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.title and "Нурлы" in at.title[0].value
    assert "result" in at.session_state and isinstance(at.session_state["result"].forecast, pd.DataFrame)
    assert at.session_state["workspace"] == "Оператор" and at.chat_input      # карточка агента с диалогом наверху
    at.session_state["workspace"] = "Исследование площадки"                  # второе рабочее пространство: карта и форма
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    labels = [b.label for b in at.button]
    assert "Исследовать площадку" in labels and "Вернуться к «Нурлы»" in labels


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
