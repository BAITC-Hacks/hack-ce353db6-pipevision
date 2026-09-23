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
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"      # реанализ ERA5 — оценка ресурса новой площадки
POWER_HOURLY_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"   # запасной источник: NASA POWER (MERRA-2)
POWER_SHEAR_ALPHA = 0.143                                          # степенной закон 50 м → 100 м для ветра MERRA-2


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
    @staticmethod
    def _coords(turbine: str, coords: tuple[float, float] | None) -> tuple[float, float]:
        """Координаты: явно переданные (любая площадка) либо турбины «Нурлы» из config.TURBINES."""
        if coords is not None:
            return float(coords[0]), float(coords[1])
        return config.TURBINES[turbine]

    def previous_runs(self, turbine: str, start: str, end: str, lead_days=config.LEAD_DAYS,
                      model: str = "best_match", coords: tuple[float, float] | None = None,
                      cache_key: str | None = None) -> pd.DataFrame:
        """Архивные прогнозы: колонки var, var_previous_day1, var_previous_day2 (UTC, почасово).

        model="best_match" — полный набор WEATHER_VARS; для отдельных моделей ансамбля — только ENSEMBLE_VARS.
        coords/cache_key — для произвольной площадки (иначе берётся турбина «Нурлы»).
        """
        lat, lon = self._coords(turbine, coords)
        variables = config.WEATHER_VARS if model == "best_match" else config.ENSEMBLE_VARS
        hourly = []
        for v in variables:
            hourly += [v] + [f"{v}_previous_day{d}" for d in lead_days]
        params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                      hourly=",".join(hourly), timezone="UTC", wind_speed_unit="ms")
        kind = "prevruns" if model == "best_match" else f"prevruns-{model}"
        if model != "best_match":
            params["models"] = model

        return self._cached_range(kind, cache_key or turbine, start, end, lambda: self._get(PREVIOUS_RUNS_URL, params))

    def ensemble_previous_runs(self, turbine: str, start: str, end: str, coords: tuple[float, float] | None = None,
                               cache_key: str | None = None) -> dict[str, pd.DataFrame]:
        """Архивные прогнозы каждой модели ансамбля; модель, которая недоступна, просто пропускается."""
        out = {}
        for m in config.ENSEMBLE_MODELS:
            try:
                out[m] = self.previous_runs(turbine, start, end, model=m, coords=coords, cache_key=cache_key)
            except Exception as e:  # noqa: BLE001 — ансамбль опционален, работаем на best_match
                log.warning("модель %s недоступна (%s), пропускаем", m, e)
        return out

    def historical_forecast(self, turbine: str, start: str, end: str) -> pd.DataFrame:
        """Прогноз с лагом 0 (склейка свежих запусков) — для истории до 16.02.2024."""
        lat, lon = config.TURBINES[turbine]

        def fetch():
            return self._get(HISTORICAL_FORECAST_URL, dict(
                latitude=lat, longitude=lon, start_date=start, end_date=end,
                hourly=",".join(config.WEATHER_VARS), timezone="UTC", wind_speed_unit="ms"))

        return self._cached_range("histforecast", turbine, start, end, fetch)

    def live_forecast(self, turbine: str, forecast_days: int = 3, coords: tuple[float, float] | None = None) -> pd.DataFrame:
        """Оперативный прогноз последнего запуска (без кэша — он меняется каждые несколько часов).

        Колонки: WEATHER_VARS (best_match) + `{model}_{var}` для моделей ансамбля (если доступны).
        """
        lat, lon = self._coords(turbine, coords)
        base = dict(latitude=lat, longitude=lon, forecast_days=forecast_days, past_days=1,
                    timezone="UTC", wind_speed_unit="ms")
        df = self._get(FORECAST_URL, {**base, "hourly": ",".join(config.WEATHER_VARS)}).set_index("time")
        for m in config.ENSEMBLE_MODELS:
            try:
                e = self._get(FORECAST_URL, {**base, "hourly": ",".join(config.ENSEMBLE_VARS), "models": m}).set_index("time")
                for v in config.ENSEMBLE_VARS:
                    df[f"{m}_{v}"] = e[v].reindex(df.index)
            except Exception as e:  # noqa: BLE001
                log.warning("live: модель %s недоступна (%s)", m, e)
        return df.reset_index()

    def archive_hourly(self, lat: float, lon: float, start: str, end: str,
                       variables=("wind_speed_100m", "wind_direction_100m", "temperature_2m", "surface_pressure"),
                       cache_key: str | None = None) -> pd.DataFrame:
        """Реанализ ERA5 (Archive API) в произвольной точке: почасово, индекс UTC, колонки — variables.

        Нужен для оценки новой площадки (ресурс ветра за год и больше); кэш — data/cache/openmeteo/era5__<ключ>__*.csv.
        """
        lat, lon = float(lat), float(lon)
        variables = list(variables)
        default = ["wind_speed_100m", "wind_direction_100m", "temperature_2m", "surface_pressure"]
        kind = "era5" if variables == default else "era5-" + "-".join(v.replace("_", "") for v in variables)
        key = cache_key or f"{lat:.3f}_{lon:.3f}"
        params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end, hourly=",".join(variables),
                      models="era5", timezone="UTC", wind_speed_unit="ms")
        source = "era5"
        try:
            df = self._cached_range(kind, key, start, end, lambda: self._get(ARCHIVE_URL, params))
        except Exception as e:  # noqa: BLE001 — лимит/недоступность Open-Meteo: запасной реанализ NASA POWER
            if self.offline or variables != default:
                raise
            log.warning("ERA5 (Open-Meteo) недоступен: %s — беру NASA POWER (MERRA-2)", e)
            df = self._cached_range("nasapower", key, start, end, lambda: self._nasa_power_hourly(lat, lon, start, end))
            source = "nasa_power"
        out = df.set_index("time")[variables]
        out.attrs["source"] = source
        return out

    def _nasa_power_hourly(self, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        """Запасной источник годового ряда: NASA POWER (MERRA-2, сетка 0.5°), почасово, UTC, без ключа и минутных лимитов.

        Колонки приводятся к контракту archive_hourly: ветер 50 м → 100 м степенным законом, давление кПа → гПа.
        """
        if self.offline:
            raise RuntimeError("offline mode: нет кэша NASA POWER для %.3f, %.3f" % (lat, lon))
        self.calls += 1
        params = {"parameters": "WS50M,WD50M,T2M,PS", "community": "RE", "longitude": lon, "latitude": lat,
                  "start": start.replace("-", ""), "end": end.replace("-", ""), "format": "JSON", "time-standard": "UTC"}
        r = requests.get(POWER_HOURLY_URL, params=params, timeout=max(self.timeout, 120))
        r.raise_for_status()
        p = r.json()["properties"]["parameter"]
        keys = sorted(p["WS50M"])
        df = pd.DataFrame({"time": pd.to_datetime(keys, format="%Y%m%d%H", utc=True),
                           "wind_speed_100m": [p["WS50M"][k] for k in keys],
                           "wind_direction_100m": [p["WD50M"][k] for k in keys],
                           "temperature_2m": [p["T2M"][k] for k in keys],
                           "surface_pressure": [p["PS"][k] for k in keys]})
        df = df.replace(-999.0, float("nan"))
        df["wind_speed_100m"] = df["wind_speed_100m"] * (100.0 / 50.0) ** POWER_SHEAR_ALPHA
        df["surface_pressure"] = df["surface_pressure"] * 10.0
        df.attrs["grid"] = (lat, lon, None)
        return df

    def prefetch(self, start: str = "2026-01-30", end: str = "2026-03-02") -> None:
        """Один запрос на весь тестовый период для каждой турбины и модели, чтобы не плодить мелкие файлы кэша."""
        for t in config.TURBINES:
            self.previous_runs(t, start, end)
            self.ensemble_previous_runs(t, start, end)

    def archived_forecast(self, turbine: str, issue_date: str, coords: tuple[float, float] | None = None,
                          cache_key: str | None = None, issue_hour: int | None = None) -> pd.DataFrame:
        """Прогноз, каким он был в момент выпуска (день `issue_date`, час `issue_hour` местного времени,
        по умолчанию 23:00), на следующие 48 часов.

        Для целевого часа t берём запуск с лагом N = ceil((t − выпуск)/24 ч) суток (`previous_dayN`):
        это самый свежий архивный запуск, который уже был доступен на момент выпуска. При выпуске в 23:00
        сутки D+1 идут из запуска за 1 сутки, D+2 — за 2 суток.
        Возвращает почасовой DataFrame (UTC) с колонками WEATHER_VARS (+ ансамбль) + lead_hours + lead_day + target_local.
        """
        from .data import local_to_utc  # локальный импорт, чтобы избежать цикла

        issue_hour = config.ISSUE_HOUR_LOCAL if issue_hour is None else int(issue_hour)
        issue_local = pd.Timestamp(issue_date).normalize() + pd.Timedelta(hours=issue_hour)
        targets_local = pd.date_range(issue_local + pd.Timedelta(hours=1), periods=config.HORIZON_HOURS, freq="h")
        targets_utc = local_to_utc(pd.DatetimeIndex(targets_local))
        start = (targets_utc.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (targets_utc.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = self.previous_runs(turbine, start, end, coords=coords, cache_key=cache_key).set_index("time")
        ens = {m: d.set_index("time") for m, d in self.ensemble_previous_runs(turbine, start, end, coords=coords, cache_key=cache_key).items()}
        rows = []
        for h, (tl, tu) in enumerate(zip(targets_local, targets_utc), start=1):
            lead_day = (h + 23) // 24                                             # 1 для часов 1–24, 2 для 25–48
            rec = {v: raw.at[tu, f"{v}_previous_day{lead_day}"] for v in config.WEATHER_VARS}
            for m, d in ens.items():
                for v in config.ENSEMBLE_VARS:
                    col = f"{v}_previous_day{lead_day}"
                    rec[f"{m}_{v}"] = d.at[tu, col] if tu in d.index and col in d else float("nan")
            rec.update(time=tu, target_local=tl, lead_day=lead_day)
            rows.append(rec)
        out = pd.DataFrame(rows)
        issue_utc = local_to_utc(pd.DatetimeIndex([issue_local]))[0]
        out["lead_hours"] = ((out["time"] - issue_utc) / pd.Timedelta(hours=1)).astype(int)
        out["issue_time_utc"] = issue_utc
        return out
