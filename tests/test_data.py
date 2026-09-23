"""Перевод времени датасета (UTC+6 до 01.03.2024, UTC+5 после) и часовая агрегация SCADA."""
from __future__ import annotations

import pandas as pd

from wind_agent.data import load_hourly, local_to_utc, utc_to_local


def test_local_to_utc_before_switch_is_utc_plus_6():
    out = local_to_utc(pd.DatetimeIndex(["2024-02-29 12:00"]))
    assert out[0] == pd.Timestamp("2024-02-29 06:00", tz="UTC")


def test_local_to_utc_after_switch_is_utc_plus_5():
    out = local_to_utc(pd.DatetimeIndex(["2024-03-01 12:00"]))
    assert out[0] == pd.Timestamp("2024-03-01 07:00", tz="UTC")


def test_utc_to_local_inverse():
    utc = pd.DatetimeIndex(["2024-02-29 06:00", "2024-03-01 07:00"]).tz_localize("UTC")
    local = utc_to_local(utc)
    assert list(local) == [pd.Timestamp("2024-02-29 12:00"), pd.Timestamp("2024-03-01 12:00")]


def test_round_trip_around_switch():
    local = pd.date_range("2024-02-28 00:00", "2024-03-02 23:00", freq="h")
    assert (utc_to_local(local_to_utc(local)) == local).all()


def test_load_hourly_t1():
    h = load_hourly("t1")
    assert isinstance(h.index, pd.DatetimeIndex)
    assert str(h.index.tz) == "UTC"
    assert h.index.is_monotonic_increasing and h.index.is_unique
    assert h["power"].between(0, 1).all()
    # первая запись датасета 2023-03-11 00:00 местного (UTC+6) → 2023-03-10 18:00 UTC
    assert h.index.min() == pd.Timestamp("2023-03-10 18:00", tz="UTC")
    # последняя 2026-01-31 23:xx местного (UTC+5) → 2026-01-31 18:00 UTC
    assert h.index.max() == pd.Timestamp("2026-01-31 18:00", tz="UTC")
    assert {"power", "wind_meas", "temp_meas", "n", "curtailed_frac"} <= set(h.columns)
