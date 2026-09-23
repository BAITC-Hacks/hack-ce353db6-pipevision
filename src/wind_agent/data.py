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


def weather_neighbors(df: pd.DataFrame) -> pd.DataFrame:
    """Exact adjacent NWP hours inside one turbine, forecast release and lead.

    Without an explicit issue timestamp (historical training), the nominal
    23:00 release means each lead covers one local calendar day. Forecast
    frames carry issue_time_utc, including custom issue hours. Missing hours
    and release/lead edges use the current forecast; row order is irrelevant.
    """
    times = pd.DatetimeIndex(df.index).as_unit("ns")
    turbine = df["turbine"].astype(str).to_numpy() if "turbine" in df else np.repeat("single", len(df))
    lead = df["lead_day"].to_numpy()
    release = (pd.DatetimeIndex(pd.to_datetime(df["issue_time_utc"], utc=True)).asi8
               if "issue_time_utc" in df else utc_to_local(times).normalize().asi8)
    key = pd.MultiIndex.from_arrays([turbine, lead, release, times.asi8])
    if key.has_duplicates:
        raise ValueError("duplicate weather hour within turbine, issue and lead")
    current = df["wind_speed_100m"].astype(float).to_numpy()
    source = pd.Series(current, index=key)
    context = pd.DataFrame(index=df.index)
    hour = pd.Timedelta(hours=1).value
    for name, offset in [("ws100_prev", -1), ("ws100_next", 1)]:
        target = pd.MultiIndex.from_arrays([turbine, lead, release, times.asi8 + offset * hour])
        adjacent = source.reindex(target).to_numpy()
        # Kazakhstan repeated local 23:00 on the UTC+6 -> UTC+5 transition.
        # The serving local-time grid has no second 23:00; do not use it as +1h.
        local_adjacent = utc_to_local(times + pd.Timedelta(hours=offset)) == (utc_to_local(times) + pd.Timedelta(hours=offset))
        context[name] = np.where(pd.notna(adjacent) & local_adjacent, adjacent, current)
    return context


def training_frame(turbine: str, wx: WeatherClient, start: str = config.PREVIOUS_RUNS_START,
                   end: str = config.HISTORY_END) -> pd.DataFrame:
    """Обучающая выборка: факт мощности + прогноз с честным лагом (previous_day1 и previous_day2).

    Каждый час истории входит дважды: с прогнозом за 1 сутки (lead_day=1) и за 2 суток (lead_day=2).
    Это учит модель тому же типу входа, который она получит в тесте.
    """
    scada = load_hourly(turbine)
    raw = wx.previous_runs(turbine, start, end).set_index("time")
    ens = {m: d.set_index("time") for m, d in wx.ensemble_previous_runs(turbine, start, end).items()}
    parts = []
    for lead in config.LEAD_DAYS:
        cols = {f"{v}_previous_day{lead}": v for v in config.WEATHER_VARS}
        f = raw[list(cols)].rename(columns=cols)
        f["lead_day"] = lead
        for m, d in ens.items():
            for v in config.ENSEMBLE_VARS:
                f[f"{m}_{v}"] = d[f"{v}_previous_day{lead}"].reindex(f.index)
        # Forecast-only context must be computed before filtering/joining SCADA.
        # Otherwise a missing/curtailed actual hour changes the weather features.
        f[["ws100_prev", "ws100_next"]] = weather_neighbors(f)
        parts.append(f)
    feats = pd.concat(parts)
    df = feats.join(scada, how="inner")
    ok = (df["n"] >= config.MIN_SAMPLES_PER_HOUR) & (df["curtailed_frac"] == 0) & df["wind_speed_100m"].notna()
    log.info("%s: обучающих строк %d (отброшено %d: мало замеров/простои/нет прогноза)", turbine, ok.sum(), (~ok).sum())
    df = df[ok].copy()
    df["turbine"] = turbine
    return df
