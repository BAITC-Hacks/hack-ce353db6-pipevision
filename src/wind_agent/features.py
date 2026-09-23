"""Признаки для модели «прогноз погоды → нормированная мощность»."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import utc_to_local

R_DRY_AIR = 287.05  # Дж/(кг·К)

ENSEMBLE_FEATURES = [f"{m}_{v}" for m in config.ENSEMBLE_MODELS for v in ("wind_speed_100m", "wind_speed_10m")] + [
    "ens_mean_ws100", "ens_std_ws100", "ens_min_ws100", "ens_max_ws100",
]

FEATURES = [
    "wind_speed_10m", "wind_speed_80m", "wind_speed_100m", "wind_speed_120m", "wind_gusts_10m",
    "ws100_prev", "ws100_next", "ws100_cubed", "shear", "gust_ratio",
    "dir_sin", "dir_cos", "temperature_2m", "surface_pressure", "air_density",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "lead_day", "turbine_id",
] + ENSEMBLE_FEATURES


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: индекс UTC (почасово), колонки WEATHER_VARS + lead_day + turbine. Возвращает матрицу FEATURES.

    ws100_prev/next — прогноз соседних часов: часовое среднее SCADA центрировано на HH:25,
    а значения Open-Meteo — на HH:00, соседние часы позволяют модели учесть это смещение.
    """
    x = pd.DataFrame(index=df.index)
    for v in ["wind_speed_10m", "wind_speed_80m", "wind_speed_100m", "wind_speed_120m", "wind_gusts_10m",
              "temperature_2m", "surface_pressure"]:
        x[v] = df[v].astype(float)
    # соседние часы внутри одной серии (одна турбина, один лаг), без утечки между сериями
    grp = [df["turbine"].values, df["lead_day"].values] if "turbine" in df else [df["lead_day"].values]
    ws = df["wind_speed_100m"].astype(float)
    x["ws100_prev"] = ws.groupby(grp).shift(1).fillna(ws)
    x["ws100_next"] = ws.groupby(grp).shift(-1).fillna(ws)
    x["ws100_cubed"] = ws ** 3
    x["shear"] = (df["wind_speed_120m"] / df["wind_speed_10m"].clip(lower=0.5)).clip(upper=10)
    x["gust_ratio"] = (df["wind_gusts_10m"] / df["wind_speed_10m"].clip(lower=0.5)).clip(upper=10)
    rad = np.deg2rad(df["wind_direction_100m"].astype(float))
    x["dir_sin"], x["dir_cos"] = np.sin(rad), np.cos(rad)
    x["air_density"] = (df["surface_pressure"] * 100.0) / (R_DRY_AIR * (df["temperature_2m"] + 273.15))
    local = utc_to_local(df.index)
    x["hour_sin"], x["hour_cos"] = np.sin(2 * np.pi * local.hour / 24), np.cos(2 * np.pi * local.hour / 24)
    x["doy_sin"], x["doy_cos"] = np.sin(2 * np.pi * local.dayofyear / 365.25), np.cos(2 * np.pi * local.dayofyear / 365.25)
    x["lead_day"] = df["lead_day"].astype(int)
    # номер турбины: t1/t2 площадки «Нурлы»; для чужой площадки — TRANSFER_TURBINE_ID (перенос кривой мощности)
    idx = {k: i for i, k in enumerate(config.TURBINES)}
    x["turbine_id"] = df["turbine"].map(lambda k: idx.get(k, config.TRANSFER_TURBINE_ID)).astype(int) if "turbine" in df else 0
    # ансамбль моделей погоды: если колонки модели нет (недоступна) — подставляем best_match, чтобы прогноз не падал
    members = []
    for m in config.ENSEMBLE_MODELS:
        for v in ("wind_speed_100m", "wind_speed_10m"):
            col = f"{m}_{v}"
            x[col] = df[col].astype(float).fillna(df[v].astype(float)) if col in df else df[v].astype(float)
        members.append(x[f"{m}_wind_speed_100m"])
    members.append(ws)
    ens = pd.concat(members, axis=1)
    x["ens_mean_ws100"], x["ens_std_ws100"] = ens.mean(axis=1), ens.std(axis=1).fillna(0.0)
    x["ens_min_ws100"], x["ens_max_ws100"] = ens.min(axis=1), ens.max(axis=1)
    return x[FEATURES]
