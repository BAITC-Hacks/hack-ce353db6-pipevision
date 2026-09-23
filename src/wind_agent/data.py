"""Загрузка SCADA-данных турбин, часовая агрегация, перевод местного времени в UTC, стыковка с прогнозами."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .weather import WeatherClient

log = logging.getLogger(__name__)

RAW_COLUMNS = ["id", "time_local", "wind_meas", "power", "temp_meas"]


def local_to_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Местное время датасета → UTC (UTC+6 до 01.03.2024, UTC+5 после)."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    offset = np.where(idx < pd.Timestamp(config.TZ_SWITCH_LOCAL), config.UTC_OFFSET_BEFORE, config.UTC_OFFSET_AFTER)
    return (idx - pd.to_timedelta(offset, unit="h")).tz_localize("UTC")


def utc_to_local(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC → местное время (для вывода прогнозов; после 01.03.2024 всегда UTC+5)."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    switch_utc = pd.Timestamp(config.TZ_SWITCH_LOCAL) - pd.Timedelta(hours=config.UTC_OFFSET_BEFORE)
    offset = np.where(idx < switch_utc, config.UTC_OFFSET_BEFORE, config.UTC_OFFSET_AFTER)
    return idx + pd.to_timedelta(offset, unit="h")


def load_raw(turbine: str) -> pd.DataFrame:
    df = pd.read_csv(config.RAW_FILES[turbine])
    df.columns = RAW_COLUMNS
    df["time_local"] = pd.to_datetime(df["time_local"])
    return df


def load_hourly(turbine: str) -> pd.DataFrame:
    """10-минутные данные → часовые средние (индекс UTC).

    Колонки: power (нормированная мощность), wind_meas, temp_meas, n (число 10-мин записей в часе),
    curtailed_frac (доля записей с ветром > 8 м/с и мощностью < 0.02 — простои/ограничения).
    """
    df = load_raw(turbine).set_index("time_local").sort_index()
    df["curtailed"] = ((df["wind_meas"] > config.CURTAIL_WIND_MS) & (df["power"] < config.CURTAIL_POWER)).astype(float)
    h = df[["power", "wind_meas", "temp_meas"]].resample("1h").mean()
    h["n"] = df["power"].resample("1h").count()
    h["curtailed_frac"] = df["curtailed"].resample("1h").mean()
    h = h[h["n"] > 0]
    h.index = local_to_utc(h.index)
    h.index.name = "time"
    return h


def training_frame(turbine: str, wx: WeatherClient, start: str = config.PREVIOUS_RUNS_START,
                   end: str = config.HISTORY_END) -> pd.DataFrame:
    """Обучающая выборка: факт мощности + прогноз с честным лагом (previous_day1 и previous_day2).

    Каждый час истории входит дважды: с прогнозом за 1 сутки (lead_day=1) и за 2 суток (lead_day=2).
    Это учит модель тому же типу входа, который она получит в тесте.
    """
    scada = load_hourly(turbine)
    raw = wx.previous_runs(turbine, start, end).set_index("time")
    parts = []
    for lead in config.LEAD_DAYS:
        cols = {f"{v}_previous_day{lead}": v for v in config.WEATHER_VARS}
        f = raw[list(cols)].rename(columns=cols)
        f["lead_day"] = lead
        parts.append(f)
    feats = pd.concat(parts)
    df = feats.join(scada, how="inner")
    ok = (df["n"] >= config.MIN_SAMPLES_PER_HOUR) & (df["curtailed_frac"] == 0) & df["wind_speed_100m"].notna()
    log.info("%s: обучающих строк %d (отброшено %d: мало замеров/простои/нет прогноза)", turbine, ok.sum(), (~ok).sum())
    df = df[ok].copy()
    df["turbine"] = turbine
    return df
