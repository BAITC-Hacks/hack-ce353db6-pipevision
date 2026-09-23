"""Протокол «как в прошлом»: прогноз, выпущенный в конце дня D, на D+1 (previous_day1) и D+2 (previous_day2)."""
from __future__ import annotations

import pandas as pd
import pytest

from wind_agent import config
from wind_agent.weather import WeatherClient


def test_archived_forecast_shape_and_leads(archived_t1):
    fc = archived_t1
    assert len(fc) == config.HORIZON_HOURS == 48
    assert fc["lead_hours"].tolist() == list(range(1, 49))
    assert (fc["lead_day"].iloc[:24] == 1).all()
    assert (fc["lead_day"].iloc[24:] == 2).all()


def test_archived_forecast_has_no_nan(archived_t1):
    assert not archived_t1[config.WEATHER_VARS].isna().any().any()


def test_archived_forecast_targets_start_next_local_day(archived_t1):
    fc = archived_t1
    assert fc["target_local"].iloc[0] == pd.Timestamp("2026-02-11 00:00")
    assert fc["target_local"].iloc[-1] == pd.Timestamp("2026-02-12 23:00")
    # выпуск в 23:00 местного (UTC+5) дня D = 18:00 UTC
    assert (fc["issue_time_utc"] == pd.Timestamp("2026-02-10 18:00", tz="UTC")).all()
    assert fc["time"].iloc[0] == pd.Timestamp("2026-02-10 19:00", tz="UTC")


def test_offline_without_cache_raises(tmp_path):
    wx = WeatherClient(cache_dir=tmp_path, offline=True)
    with pytest.raises(RuntimeError, match="offline"):
        wx.archived_forecast("t1", "2026-02-10")
