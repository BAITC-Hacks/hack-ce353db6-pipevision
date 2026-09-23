"""Признаки для модели «прогноз погоды → нормированная мощность»."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import utc_to_local

R_DRY_AIR = 287.05  # Дж/(кг·К)

FEATURES = [
    "wind_speed_10m", "wind_speed_80m", "wind_speed_100m", "wind_speed_120m", "wind_gusts_10m",
    "ws100_prev", "ws100_next", "ws100_cubed", "shear", "gust_ratio",
    "dir_sin", "dir_cos", "temperature_2m", "surface_pressure", "air_density",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "lead_day", "turbine_id",
]


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
    x["turbine_id"] = df["turbine"].map({k: i for i, k in enumerate(config.TURBINES)}).astype(int) if "turbine" in df else 0
    return x[FEATURES]
