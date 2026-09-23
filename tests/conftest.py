"""Общие фикстуры. Все тесты работают офлайн: данные из data/raw, погода из кэша data/cache/openmeteo."""
from __future__ import annotations

import pandas as pd
import pytest

from wind_agent.weather import WeatherClient

ISSUE_DATE = "2026-02-10"  # день выпуска внутри тестового февраля


@pytest.fixture(scope="session")
def wx() -> WeatherClient:
    return WeatherClient(offline=True)


@pytest.fixture(scope="session")
def archived_t1(wx) -> pd.DataFrame:
    """Архивный прогноз погоды для t1, каким он был в конце дня 2026-02-10."""
    return wx.archived_forecast("t1", ISSUE_DATE)


def model_input(fc: pd.DataFrame, turbine: str) -> pd.DataFrame:
    """Вывод archived_forecast → вход make_features (индекс UTC + колонка turbine)."""
    df = fc.set_index("time").copy()
    df["turbine"] = turbine
    return df
