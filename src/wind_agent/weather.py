"""Клиент Open-Meteo с дисковым кэшем.

Три источника:
* Previous Runs API  — архив прогнозов: значение `X_previous_dayN` для часа t взято из запуска модели,
  сделанного за N суток до t. Именно так воспроизводится «прогноз, доступный на момент выпуска».
* Historical Forecast API — свежие запуски (лаг 0), нужен для истории до 16.02.2024.
* Forecast API — текущий оперативный прогноз для режима live.

Все ответы кладём в data/cache/openmeteo/*.csv, чтобы жюри могло воспроизвести прогон офлайн.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherClient:
    def __init__(self, cache_dir: Path = config.CACHE_DIR, offline: bool = False, timeout: int = 120):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.timeout = timeout
        self.calls = 0

    # ---------------------------------------------------------------- low level
    def _get(self, url: str, params: dict) -> pd.DataFrame:
        if self.offline:
            raise RuntimeError("offline mode: нет кэша для запроса %s %s" % (url, params))
        self.calls += 1
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        js = r.json()
        df = pd.DataFrame(js["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.attrs["grid"] = (js.get("latitude"), js.get("longitude"), js.get("elevation"))
        return df

    def _cached_range(self, kind: str, turbine: str, start: str, end: str, fetch) -> pd.DataFrame:
        """Ищем в кэше файл этого типа/турбины, покрывающий [start, end]; иначе скачиваем и сохраняем."""
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        for f in sorted(self.cache_dir.glob(f"{kind}__{turbine}__*.csv")):
            _, _, s, e = f.stem.split("__")
            if pd.Timestamp(s) <= start_ts and pd.Timestamp(e) >= end_ts:
                df = pd.read_csv(f, parse_dates=["time"])
                df["time"] = pd.to_datetime(df["time"], utc=True)
                return df[(df["time"] >= start_ts.tz_localize("UTC")) & (df["time"] < end_ts.tz_localize("UTC") + pd.Timedelta(days=1))]
        df = fetch()
        out = self.cache_dir / f"{kind}__{turbine}__{start}__{end}.csv"
        df.to_csv(out, index=False)
        log.info("Open-Meteo %s %s %s..%s → %s строк, кэш %s", kind, turbine, start, end, len(df), out.name)
        return df

    # ---------------------------------------------------------------- public
    def previous_runs(self, turbine: str, start: str, end: str, lead_days=config.LEAD_DAYS) -> pd.DataFrame:
        """Архивные прогнозы: колонки var, var_previous_day1, var_previous_day2 (UTC, почасово)."""
        lat, lon = config.TURBINES[turbine]
        hourly = []
        for v in config.WEATHER_VARS:
            hourly += [v] + [f"{v}_previous_day{d}" for d in lead_days]

        def fetch():
            return self._get(PREVIOUS_RUNS_URL, dict(
                latitude=lat, longitude=lon, start_date=start, end_date=end,
                hourly=",".join(hourly), timezone="UTC", wind_speed_unit="ms"))

        return self._cached_range("prevruns", turbine, start, end, fetch)

    def historical_forecast(self, turbine: str, start: str, end: str) -> pd.DataFrame:
        """Прогноз с лагом 0 (склейка свежих запусков) — для истории до 16.02.2024."""
        lat, lon = config.TURBINES[turbine]

        def fetch():
            return self._get(HISTORICAL_FORECAST_URL, dict(
                latitude=lat, longitude=lon, start_date=start, end_date=end,
                hourly=",".join(config.WEATHER_VARS), timezone="UTC", wind_speed_unit="ms"))

        return self._cached_range("histforecast", turbine, start, end, fetch)

    def live_forecast(self, turbine: str, forecast_days: int = 3) -> pd.DataFrame:
        """Оперативный прогноз последнего запуска (без кэша — он меняется каждые несколько часов)."""
        lat, lon = config.TURBINES[turbine]
        df = self._get(FORECAST_URL, dict(
            latitude=lat, longitude=lon, forecast_days=forecast_days, past_days=1,
            hourly=",".join(config.WEATHER_VARS), timezone="UTC", wind_speed_unit="ms"))
        return df

    def prefetch(self, start: str = "2026-01-30", end: str = "2026-03-02") -> None:
        """Один запрос на весь тестовый период для каждой турбины, чтобы не плодить мелкие файлы кэша."""
        for t in config.TURBINES:
            self.previous_runs(t, start, end)

    def archived_forecast(self, turbine: str, issue_date: str) -> pd.DataFrame:
        """Прогноз, каким он был в конце дня `issue_date` (местное время) на следующие 48 часов.

        Часы дня D+1 берём из запуска за 1 сутки (previous_day1), часы дня D+2 — из запуска за 2 суток
        (previous_day2). Оба запуска сделаны в день D, то есть были доступны на момент выпуска.
        Возвращает почасовой DataFrame (UTC) с колонками WEATHER_VARS + lead_hours + target_local.
        """
        from .data import local_to_utc  # локальный импорт, чтобы избежать цикла

        issue = pd.Timestamp(issue_date)
        t0_local = issue.normalize() + pd.Timedelta(days=1)                      # D+1 00:00 местного
        targets_local = pd.date_range(t0_local, periods=config.HORIZON_HOURS, freq="h")
        targets_utc = local_to_utc(pd.DatetimeIndex(targets_local))
        start = (targets_utc.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (targets_utc.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = self.previous_runs(turbine, start, end).set_index("time")
        rows = []
        for tl, tu in zip(targets_local, targets_utc):
            lead_day = (tl.normalize() - t0_local).days + 1                       # 1 для D+1, 2 для D+2
            rec = {v: raw.at[tu, f"{v}_previous_day{lead_day}"] for v in config.WEATHER_VARS}
            rec.update(time=tu, target_local=tl, lead_day=lead_day)
            rows.append(rec)
        out = pd.DataFrame(rows)
        issue_utc = local_to_utc(pd.DatetimeIndex([issue.normalize() + pd.Timedelta(hours=config.ISSUE_HOUR_LOCAL)]))[0]
        out["lead_hours"] = ((out["time"] - issue_utc) / pd.Timedelta(hours=1)).astype(int)
        out["issue_time_utc"] = issue_utc
        return out
