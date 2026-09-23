"""Оценка новой площадки ВЭС: ветровой ресурс по ERA5, ожидаемая выработка и отчёт для руководства.

Поток: координаты → почасовой ERA5 за период (Open-Meteo Archive, кэш) → статистика ветра (Weibull, роза, сезонность,
суточный ход, плотность воздуха) → КИУМ и годовая выработка по кривой мощности «Нурлы» → сравнение с «Нурлы»
(факт SCADA и ERA5 в точке «Нурлы» за тот же период) и с атласом NASA POWER → отчёт (шаблон + LLM с проверкой чисел).

Кривые мощности эмпирические, из SCADA «Нурлы» (бины 0.5 м/с): базовая — кривая турбины (ветер анемометра → мощность),
верхний сценарий — кривая по прогнозному ветру 100 м (как model.power_curve_baseline), включающая местное усиление
ветра «Нурлы» относительно моделей погоды. Проверка метода на самой «Нурлы» — в benchmark и warnings.
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import atlas, config
from .weather import WeatherClient

log = logging.getLogger(__name__)

POWER_CURVE_JSON = config.MODELS_DIR / "power_curve_nurly.json"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
RHO_STD = 1.225                      # стандартная плотность воздуха, кг/м³
R_DRY = 287.05                       # газовая постоянная сухого воздуха, Дж/(кг·К)
CUT_IN, RATED_WS, CUT_OUT = 3.0, 12.0, 25.0    # параметрическая кривая (fallback) и пороги режимов
CALM_WS, STORM_WS = 3.0, 25.0
LOCAL_UTC_OFFSET = 5                 # Казахстан — единое время UTC+5
HOURS_PER_YEAR = 8760
MIN_BIN_SAMPLES = 20
SECTOR_NAMES = ["С", "ССВ", "СВ", "ВСВ", "В", "ВЮВ", "ЮВ", "ЮЮВ",
                "Ю", "ЮЮЗ", "ЮЗ", "ЗЮЗ", "З", "ЗСЗ", "СЗ", "ССЗ"]
MONTH_RU = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август",
            "сентябрь", "октябрь", "ноябрь", "декабрь"]
_NURLY_POWER: pd.Series | None = None


# ---------------------------------------------------------------- кривая мощности
def parametric_curve() -> dict:
    """Стандартная кривая: 0 до 3 м/с, кубический рост до номинала на 12 м/с, отключение на 25 м/с."""
    ws = np.arange(0, 30.5, 0.5)
    p = np.clip((ws ** 3 - CUT_IN ** 3) / (RATED_WS ** 3 - CUT_IN ** 3), 0, 1)
    p[ws < CUT_IN] = 0
    p[ws >= CUT_OUT] = 0
    return {"ws": ws.round(2).tolist(), "power": p.round(4).tolist(),
            "source": "параметрическая кривая (cut-in 3, номинал 12, cut-out 25 м/с): данных «Нурлы» нет"}


def _binned_curve(ws: pd.Series, power: pd.Series) -> dict:
    """Средняя мощность в бинах 0.5 м/с → монотонная кривая на сетке 0…30 м/с (плато за краем, 0 после cut-out)."""
    bins = np.arange(0, 45, 0.5)
    g = power.groupby(pd.cut(ws, bins), observed=False).agg(["mean", "count"])
    centers = bins[:-1] + 0.25
    ok = (g["count"].to_numpy() >= MIN_BIN_SAMPLES) & np.isfinite(g["mean"].to_numpy())
    x, y = centers[ok], np.maximum.accumulate(np.clip(g["mean"].to_numpy()[ok], 0, 1))
    grid = np.arange(0, 30.5, 0.5)
    curve = np.interp(grid, np.r_[0.0, x], np.r_[0.0, y])
    curve[grid >= CUT_OUT] = 0
    return {"ws": grid.round(2).tolist(), "power": curve.round(4).tolist()}


def build_power_curve(wx: WeatherClient | None = None) -> dict:
    """Две эмпирические кривые «Нурлы» (t1+t2, без простоев и неполных часов, бины 0.5 м/с):

    * turbine — физическая кривая турбины: ветер анемометра SCADA → мощность. Основной расчёт: применяется
      к ветру ERA5 новой площадки как к «настоящему» ветру (для равнинной степи ERA5 близок к нему);
    * nurly_calibrated — прогнозный ветер 100 м Open-Meteo → факт мощности (та же, что model.power_curve_baseline).
      Включает местное усиление ветра «Нурлы» относительно моделей погоды — верхний сценарий.
    """
    from .data import load_hourly, training_frame
    wx = wx or WeatherClient(offline=True)
    sc = pd.concat([load_hourly(t) for t in config.TURBINES])
    sc = sc[(sc["n"] >= config.MIN_SAMPLES_PER_HOUR) & (sc["curtailed_frac"] == 0)]
    tf = pd.concat([training_frame(t, wx) for t in config.TURBINES])
    turbine = _binned_curve(sc["wind_meas"], sc["power"])
    turbine["source"] = (f"кривая турбины «Нурлы» по SCADA: ветер анемометра → мощность t1+t2, бины 0.5 м/с, "
                         f"{len(sc)} часов")
    calibrated = _binned_curve(tf["wind_speed_100m"], tf["power"])
    calibrated["source"] = (f"кривая, откалиброванная на «Нурлы»: прогнозный ветер 100 м (Open-Meteo) → факт мощности, "
                            f"{len(tf)} часов {config.PREVIOUS_RUNS_START}…{config.HISTORY_END}")
    return {"version": 2, "turbine": turbine, "nurly_calibrated": calibrated}


def power_curve(wx: WeatherClient | None = None) -> dict:
    """Кривые из models/power_curve_nurly.json ({turbine, nurly_calibrated}); нет файла — строим и записываем;
    нет данных — параметрическая кривая в обоих ролях."""
    if POWER_CURVE_JSON.exists():
        c = json.loads(POWER_CURVE_JSON.read_text(encoding="utf-8"))
        if c.get("version") == 2:
            return c
    try:
        c = build_power_curve(wx)
        POWER_CURVE_JSON.parent.mkdir(parents=True, exist_ok=True)
        POWER_CURVE_JSON.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
        return c
    except Exception as e:  # noqa: BLE001 — нет SCADA/кэша погоды: стандартная кривая
        log.warning("кривая «Нурлы» недоступна (%s), берём параметрическую", e)
        pc = parametric_curve()
        return {"version": 2, "turbine": pc, "nurly_calibrated": pc}


def apply_curve(curve: dict, ws: np.ndarray) -> np.ndarray:
    """Доля номинала по кривой; выше cut-out — 0."""
    ws = np.asarray(ws, dtype=float)
    p = np.interp(ws, curve["ws"], curve["power"])
    p[ws >= CUT_OUT] = 0
    return np.where(np.isfinite(ws), p, np.nan)


# ---------------------------------------------------------------- статистика ветра
def weibull_moments(ws: np.ndarray) -> dict:
    """Параметры Вейбулла методом моментов: k = (σ/μ)^−1.086, c = μ / Γ(1 + 1/k)."""
    ws = np.asarray(ws, dtype=float)
    ws = ws[np.isfinite(ws)]
    mu, sd = float(ws.mean()), float(ws.std())
    if mu <= 0 or sd <= 0:
        return {"k": None, "c": None}
    k = (sd / mu) ** -1.086
    return {"k": round(k, 2), "c": round(mu / math.gamma(1 + 1 / k), 2)}


def air_density(t_c: pd.Series, p_hpa: pd.Series) -> pd.Series:
    """Плотность воздуха по уравнению состояния: ρ = p / (R·T)."""
    return p_hpa * 100.0 / (R_DRY * (t_c + 273.15))


def wind_rose(ws: pd.Series, wd: pd.Series) -> list[dict]:
    ok = ws.notna() & wd.notna()
    sec = (np.floor(((wd[ok].to_numpy() % 360) + 11.25) / 22.5) % 16).astype(int)
    w = ws[ok].to_numpy()
    out = []
    for i, name in enumerate(SECTOR_NAMES):
        m = sec == i
        out.append({"sector": name, "deg": i * 22.5, "share": round(float(m.mean()), 4) if len(sec) else 0.0,
                    "ws_mean": round(float(w[m].mean()), 2) if m.any() else None})
    return out


def elevation(lat: float, lon: float, wx: WeatherClient) -> float | None:
    """Высота над уровнем моря (Open-Meteo Elevation API, DEM 90 м); кэш — elevation.json рядом с кэшем погоды."""
    cache = Path(wx.cache_dir) / "elevation.json"
    key = f"{lat:.4f}_{lon:.4f}"
    try:
        store = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    except Exception:  # noqa: BLE001
        store = {}
    if key in store:
        return store[key]
    if wx.offline:
        return None
    try:
        import requests
        wx.calls += 1
        r = requests.get(ELEVATION_URL, params={"latitude": lat, "longitude": lon}, timeout=30)
        r.raise_for_status()
        val = float(r.json()["elevation"][0])
        store[key] = val
        cache.write_text(json.dumps(store, indent=1), encoding="utf-8")
        return val
    except Exception as e:  # noqa: BLE001 — лимит/недоступность Elevation API: высота ячейки атласа (MERRA-2, ~50 км)
        log.warning("высота площадки недоступна: %s — беру высоту ячейки атласа", e)
        try:
            return float(atlas.nearest_cell(lat, lon)["elevation_m"])
        except Exception:  # noqa: BLE001
            return None


def nurly_cf_actual(start: str | None = None, end: str | None = None) -> tuple[float | None, str]:
    """Фактический КИУМ «Нурлы» по SCADA (средняя часовая мощность t1 и t2, доля номинала) и период, за который он взят.

    Если SCADA покрывает ≥ 50 % часов [start, end] — КИУМ за этот период (сопоставимо с ERA5), иначе — за всю историю.
    """
    global _NURLY_POWER
    if _NURLY_POWER is None:
        try:
            from .data import load_hourly
            vals = [load_hourly(t) for t in config.TURBINES]
            _NURLY_POWER = pd.concat([v.loc[v["n"] >= config.MIN_SAMPLES_PER_HOUR, "power"] for v in vals]).sort_index()
        except Exception as e:  # noqa: BLE001
            log.warning("SCADA «Нурлы» недоступны: %s", e)
            return None, ""
    p = _NURLY_POWER
    if start and end:
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        part = p[(p.index >= s) & (p.index < e)]
        expected = len(config.TURBINES) * (e - s) / pd.Timedelta(hours=1)
        if len(part) >= 0.5 * expected:
            return float(part.mean()), f"{start}…{end}"
    return float(p.mean()), f"{p.index.min():%Y-%m-%d}…{p.index.max():%Y-%m-%d}"


# ---------------------------------------------------------------- оценка
def _r(x, n: int = 2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def assess_site(site: config.Site, wx: WeatherClient, start: str = "2025-01-01", end: str = "2025-12-31",
                progress=None) -> dict:
    """Оценка ресурса и выработки площадки по ERA5 за [start, end]. Возвращает JSON-сериализуемый dict."""
    def step(i: int, text: str) -> None:
        if progress:
            progress(i, 5, text)
        log.info("оценка %s: %s", site.key, text)

    lat, lon = next(iter(site.turbines.values()))
    warnings: list[str] = []
    step(1, "реанализ ERA5 в точке площадки")
    df = wx.archive_hourly(lat, lon, start, end)
    wind_source = df.attrs.get("source", "era5")
    if wind_source == "nasa_power":
        warnings.append("Open-Meteo (ERA5) был недоступен (лимит запросов) — годовой ряд взят из NASA POWER "
                        "(реанализ MERRA-2, сетка 0,5°, ветер 50 м пересчитан на 100 м): ошибка оценки выше, "
                        "повторите расчёт позже для уточнения по ERA5")
    ws = df["wind_speed_100m"].astype(float)
    if ws.notna().sum() < 24 * 30:
        raise ValueError(f"мало данных ERA5 для площадки: {int(ws.notna().sum())} часов")
    step(2, "высота и кривая мощности")
    elev = elevation(lat, lon, wx)
    curves = power_curve(wx)
    curve, curve_hi = curves["turbine"], curves["nurly_calibrated"]
    step(3, "эталон: ERA5 в точке «Нурлы»")
    nlat, nlon = config.TURBINES["t1"]
    rho = air_density(df["temperature_2m"], df["surface_pressure"])
    try:
        nd = wx.archive_hourly(nlat, nlon, start, end)
        nurly_ws = float(nd["wind_speed_100m"].mean())
        rho_ref = float(air_density(nd["temperature_2m"], nd["surface_pressure"]).mean())
        nurly_cf_model = float(np.nanmean(apply_curve(curve, nd["wind_speed_100m"].to_numpy())))
        nurly_cf_model_hi = float(np.nanmean(apply_curve(curve_hi, nd["wind_speed_100m"].to_numpy())))
    except Exception as e:  # noqa: BLE001
        warnings.append(f"эталон «Нурлы» по ERA5 недоступен: {e}")
        nurly_ws, rho_ref, nurly_cf_model, nurly_cf_model_hi = None, RHO_STD, None, None
    if "параметрическая" in curve["source"]:
        rho_ref = RHO_STD
    step(4, "статистика ветра и выработка")
    # поправка на плотность: эквивалентная скорость при плотности, на которой снята кривая
    ratio_h = (rho / rho_ref).fillna(1.0).clip(0.7, 1.3)
    ws_eq = ws * ratio_h ** (1 / 3)
    power = pd.Series(apply_curve(curve, ws_eq.to_numpy()), index=df.index)
    cf = float(power.mean())
    cf_hi = max(float(np.nanmean(apply_curve(curve_hi, ws_eq.to_numpy()))), cf)
    n_t = int(site.n_turbines or 1)
    rated = float(site.rated_mw) if site.rated_mw else 2.5
    if not site.rated_mw:
        warnings.append("номинал турбины не задан — принят 2.5 МВт (как у турбин «Нурлы»)")
    aep_t = rated * cf * HOURS_PER_YEAR / 1000
    local = df.index + pd.Timedelta(hours=LOCAL_UTC_OFFSET)
    monthly = []
    for m in range(1, 13):
        mask = df.index.month == m
        if mask.any():
            monthly.append({"month": m, "name_ru": MONTH_RU[m - 1], "ws100_mean": _r(ws[mask].mean()),
                            "cf": _r(power[mask].mean(), 3)})
    diurnal = [{"hour_local": h, "ws100_mean": _r(ws[local.hour == h].mean())} for h in range(24)]
    rose = wind_rose(ws, df["wind_direction_100m"].astype(float))
    prevailing = max(rose, key=lambda r: r["share"])["sector"]
    try:
        cell = atlas.nearest_cell(lat, lon)
        kz = atlas.load_atlas()["ws100_est"]
        atl = {"ws50_ann": cell["ws50_ann"], "ws100_est": cell["ws100_est"], "resource_class": cell["resource_class"],
               "cell_lat": cell["lat"], "cell_lon": cell["lon"], "distance_km": cell["distance_km"],
               "percentile_kz": round(float((kz < cell["ws100_est"]).mean()), 3),     # доля ячеек Казахстана слабее
               "kz_max_ws100": float(kz.max())}
    except Exception as e:  # noqa: BLE001
        atl = None
        warnings.append(f"атлас недоступен: {e}")
    in_kz = atlas.in_kazakhstan(lat, lon)
    if not in_kz:
        warnings.append("точка вне границы Казахстана (или у самой границы) по полигону атласа")
    nurly_act, nurly_act_period = nurly_cf_actual(start, end)
    if site.transfer:
        warnings.append("кривая мощности — турбины «Нурлы» (2.5 МВт); для другой модели турбины и площадки без замеров "
                        "оценка ориентировочная, нужна годовая мачта/лидар")
    warnings.append("ERA5 — реанализ с сеткой ~31 км: в горах и предгорьях занижает местный ветер, "
                    "рельеф и шероховатость участка не учтены")
    if nurly_cf_model is not None and nurly_act:
        warnings.append(f"проверка метода на «Нурлы»: ERA5 + кривая турбины дают КИУМ {nurly_cf_model:.3f}, "
                        f"откалиброванная кривая — {nurly_cf_model_hi:.3f}, факт SCADA {nurly_act:.3f}: в предгорьях ERA5 "
                        f"занижает ветер, базовый сценарий консервативен")
    mean_ws = float(ws.mean())
    step(5, "готово")
    return {
        "site": {"key": site.key, "name": site.name, "lat": round(lat, 5), "lon": round(lon, 5),
                 "n_turbines": n_t, "rated_mw": rated, "capacity_mw": round(n_t * rated, 2), "transfer": site.transfer,
                 "in_kazakhstan": in_kz},
        "elevation_m": _r(elev, 1) if elev is not None else None,
        "period": {"start": start, "end": end, "hours": int(ws.notna().sum())},
        "ws100": {"mean": _r(mean_ws), "median": _r(ws.median()), "p90": _r(ws.quantile(0.9)), "max": _r(ws.max())},
        "weibull": weibull_moments(ws.to_numpy()),
        "monthly": monthly,
        "diurnal": diurnal,
        "rose": rose,
        "prevailing_sector": prevailing,
        "calm_share": _r((ws < CALM_WS).mean(), 3),
        "storm_share": _r((ws > STORM_WS).mean(), 4),
        "density_ratio": _r(rho.mean() / RHO_STD, 3),
        "density_ratio_to_curve": _r(rho.mean() / rho_ref, 3),
        "cf": _r(cf, 3),
        "aep_gwh": _r(aep_t * n_t, 2),
        "aep_per_turbine_gwh": _r(aep_t, 2),
        "full_load_hours": int(round(cf * HOURS_PER_YEAR)),
        "scenarios": {"base": {"cf": _r(cf, 3), "aep_gwh": _r(aep_t * n_t, 2)},
                      "nurly_calibrated": {"cf": _r(cf_hi, 3), "aep_gwh": _r(rated * cf_hi * HOURS_PER_YEAR / 1000 * n_t, 2)}},
        "benchmark": {"nurly_cf_actual": _r(nurly_act, 3), "nurly_cf_actual_period": nurly_act_period,
                      "nurly_cf_model": _r(nurly_cf_model, 3), "nurly_cf_model_calibrated": _r(nurly_cf_model_hi, 3),
                      "nurly_ws100_mean": _r(nurly_ws),
                      "ratio_to_nurly": _r(mean_ws / nurly_ws, 2) if nurly_ws else None,
                      "cf_ratio_to_nurly_model": _r(cf / nurly_cf_model, 2) if nurly_cf_model else None},
        "atlas": atl,
        "power_curve_source": curve["source"],
        "power_curve_calibrated_source": curve_hi["source"],
        "data_sources": [
            (f"Open-Meteo Archive API (реанализ ERA5), почасово {start}…{end}: ветер и направление 100 м, T 2 м, давление"
             if wind_source == "era5" else
             f"NASA POWER hourly (реанализ MERRA-2, 0,5°), почасово {start}…{end}: ветер 50 м → 100 м, T 2 м, давление "
             "(запасной источник вместо ERA5)"),
            "Open-Meteo Elevation API (DEM Copernicus 90 м) — высота площадки",
            "NASA POWER Climatology (MERRA-2, 2001–2020) — атлас ветра Казахстана, WS50M",
            "SCADA ВЭС «Нурлы» (t1, t2) — кривая мощности и фактический КИУМ",
        ],
        "warnings": warnings,
        "computed_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_assessment(a: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "assessment.json"
    path.write_text(json.dumps(a, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------- отчёт для руководства
def _pct(x, n: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{n}f} %"


def _num(x, n: int = 2, dash: str = "—") -> str:
    return dash if x is None else f"{x:.{n}f}"


def _recommendation(a: dict) -> tuple[str, str]:
    """Вердикт по КИУМ относительно факта «Нурлы» и классу ресурса."""
    cf = a.get("cf") or 0
    ref = (a.get("benchmark") or {}).get("nurly_cf_model") or (a.get("benchmark") or {}).get("nurly_cf_actual")
    if cf >= 0.35 or (ref and cf >= 0.95 * ref):
        return "рекомендуется", "перейти к предпроектным изысканиям: установить мачту или лидар на 12 месяцев"
    if cf >= 0.25:
        return "условно рекомендуется", "уточнить ресурс мачтой/лидаром и сравнить с соседними ячейками атласа"
    return "не рекомендуется", "рассмотреть ячейки атласа с классом «хороший» или «отличный»"


def template_report(a: dict, forecast_summary: dict | None = None) -> str:
    s, b, w = a["site"], a.get("benchmark") or {}, a["ws100"]
    atl = a.get("atlas") or {}
    verdict, action = _recommendation(a)
    months = [m for m in a.get("monthly", []) if m.get("cf") is not None]
    best = max(months, key=lambda m: m["cf"]) if months else None
    worst = min(months, key=lambda m: m["cf"]) if months else None
    di = [d for d in a.get("diurnal", []) if d.get("ws100_mean") is not None]
    dmax = max(di, key=lambda d: d["ws100_mean"]) if di else None
    dmin = min(di, key=lambda d: d["ws100_mean"]) if di else None
    rose = sorted(a.get("rose", []), key=lambda r: -r["share"])[:3]
    L = [f"# Оценка площадки ВЭС: {s['name']}", ""]
    L += ["## Резюме", "",
          f"Площадка {s['lat']:.4f}, {s['lon']:.4f}: средний ветер на 100 м {_num(w['mean'])} м/с (ERA5, "
          f"{a['period']['start']} — {a['period']['end']}), ожидаемый КИУМ {_pct(a['cf'])}, годовая выработка "
          f"{_num(a['aep_gwh'])} ГВт·ч для {s['n_turbines']} турбин по {_num(s['rated_mw'], 1)} МВт "
          f"({_num(s['capacity_mw'], 1)} МВт); верхний сценарий — КИУМ {_pct(((a.get('scenarios') or {}).get('nurly_calibrated') or {}).get('cf'))}. "
          f"Вывод: **{verdict}** — {action}.", ""]
    L += ["## Площадка", "",
          f"- Координаты: {s['lat']:.4f} с.ш., {s['lon']:.4f} в.д."
          + (f"; высота {a['elevation_m']:.0f} м" if a.get("elevation_m") is not None else ""),
          f"- Состав: {s['n_turbines']} × {_num(s['rated_mw'], 1)} МВт = {_num(s['capacity_mw'], 1)} МВт",
          f"- Плотность воздуха: {_num(a['density_ratio'], 3)} от стандартной",
          ""]
    L += ["## Ветровой ресурс", "",
          f"- Средняя скорость на 100 м: {_num(w['mean'])} м/с, медиана {_num(w['median'])}, P90 {_num(w['p90'])}, "
          f"максимум {_num(w['max'])} м/с",
          f"- Распределение Вейбулла: k = {_num(a['weibull'].get('k'))}, c = {_num(a['weibull'].get('c'))} м/с",
          f"- Преобладающее направление: {a['prevailing_sector']}; топ-3 румба: "
          + ", ".join(f"{r['sector']} {_pct(r['share'])}" for r in rose),
          f"- Штиль (< 3 м/с): {_pct(a['calm_share'])} часов, шторм (> 25 м/с): {_pct(a['storm_share'], 2)}"]
    if atl:
        L.append(f"- Атлас NASA POWER (2001–2020): ветер на 50 м {_num(atl.get('ws50_ann'))} м/с, "
                 f"оценка на 100 м {_num(atl.get('ws100_est'))} м/с — класс «{atl.get('resource_class')}»"
                 + (f"; ветер выше, чем в {_pct(atl['percentile_kz'], 0)} ячеек атласа Казахстана"
                    if atl.get("percentile_kz") is not None else ""))
    L.append("")
    L += ["## Ожидаемая выработка и сравнение с «Нурлы»", "",
          f"- КИУМ: {_pct(a['cf'])}, эквивалент {a['full_load_hours']} часов работы на номинале в год",
          f"- Годовая выработка: {_num(a['aep_gwh'])} ГВт·ч, на одну турбину {_num(a['aep_per_turbine_gwh'])} ГВт·ч"]
    if b.get("nurly_cf_actual") is not None:
        L.append(f"- «Нурлы»: фактический КИУМ по SCADA {_pct(b['nurly_cf_actual'])} ({b.get('nurly_cf_actual_period', '')})"
                 + (f", по тому же методу (ERA5 + кривая) {_pct(b['nurly_cf_model'])}" if b.get("nurly_cf_model") is not None else ""))
    sc = (a.get("scenarios") or {}).get("nurly_calibrated") or {}
    if sc.get("cf") is not None:
        L.append(f"- Верхний сценарий (кривая, откалиброванная на факте «Нурлы» по прогнозному ветру): "
                 f"КИУМ {_pct(sc['cf'])}, выработка {_num(sc['aep_gwh'])} ГВт·ч в год")
    if b.get("nurly_ws100_mean") is not None:
        L.append(f"- Ветер ERA5 на 100 м в точке «Нурлы» за тот же период {_num(b['nurly_ws100_mean'])} м/с; "
                 f"отношение площадки к «Нурлы» {_num(b.get('ratio_to_nurly'))}")
    L.append(f"- Кривая мощности: {a['power_curve_source']}")
    L.append("")
    L += ["## Сезонность и режим", ""]
    if best and worst:
        L.append(f"- Лучший месяц — {best['name_ru']} (КИУМ {_pct(best['cf'])}, ветер {_num(best['ws100_mean'])} м/с), "
                 f"худший — {worst['name_ru']} (КИУМ {_pct(worst['cf'])}, ветер {_num(worst['ws100_mean'])} м/с)")
    if dmax and dmin:
        L.append(f"- Суточный ход (местное время): максимум ветра в {dmax['hour_local']} ч "
                 f"({_num(dmax['ws100_mean'])} м/с), минимум в {dmin['hour_local']} ч ({_num(dmin['ws100_mean'])} м/с)")
    if forecast_summary:
        L.append("- Оперативный прогноз модели WindAgent на 48 ч: " + "; ".join(
            f"{k} = {v}" for k, v in forecast_summary.items() if isinstance(v, (int, float, str))))
    L.append("")
    L += ["## Риски и ограничения", ""] + [f"- {x}" for x in a.get("warnings", [])] + [""]
    L += ["## Рекомендация", "",
          f"**{verdict.capitalize()}.** Следующий шаг: {action}. Оценка основана на одном году реанализа; "
          f"перед инвестиционным решением нужны замеры на площадке, анализ сетевого присоединения и ограничений землепользования.",
          "", f"_Источники: {'; '.join(a.get('data_sources', []))}. Расчёт {a.get('computed_at_utc')}._"]
    return "\n".join(L)


def _flat_numbers(obj, out: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if np.isfinite(obj):
            out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flat_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flat_numbers(v, out)
    elif isinstance(obj, str):
        from .agent.tools import extract_numbers
        out.extend(v for _, v, _ in extract_numbers(obj)[0])


def allowed_facts(a: dict, forecast_summary: dict | None = None) -> dict:
    """«Анализ» для verify_narrative: все числа оценки и производные (МВт·ч, проценты, отношения, пороги)."""
    nums: list[float] = []
    _flat_numbers(a, nums)
    if forecast_summary:
        _flat_numbers(forecast_summary, nums)
    b = a.get("benchmark") or {}
    extra = [HOURS_PER_YEAR, RHO_STD, CALM_WS, STORM_WS, CUT_IN, RATED_WS, CUT_OUT, 16, 12, 50, 100, 0.143, 5.5, 6.5, 7.5,
             LOCAL_UTC_OFFSET, 31, 90, 2001, 2020, 0.5, 1]
    if a.get("aep_gwh") is not None:
        extra += [a["aep_gwh"] * 1000, a["aep_per_turbine_gwh"] * 1000]
    if a.get("cf") is not None and b.get("nurly_cf_actual"):
        extra += [a["cf"] / b["nurly_cf_actual"], a["cf"] - b["nurly_cf_actual"]]
    if a.get("cf") is not None and b.get("nurly_cf_model"):
        extra += [a["cf"] - b["nurly_cf_model"]]
    if b.get("nurly_cf_actual") and b.get("nurly_cf_model"):
        extra += [b["nurly_cf_model"] / b["nurly_cf_actual"]]
    if b.get("ratio_to_nurly"):
        extra += [b["ratio_to_nurly"] - 1]
    if b.get("nurly_cf_actual"):
        cap = (a["site"].get("capacity_mw") or 0)
        extra += [cap * b["nurly_cf_actual"] * HOURS_PER_YEAR / 1000]
    metrics = {f"v{i}": v for i, v in enumerate(nums + extra)}
    return {"metrics": {"site": metrics}, "days": [a["period"]["start"], a["period"]["end"]], "flags": [], "series": []}


LLM_SYSTEM = """Ты — аналитик по ветроэнергетике. По JSON с результатами оценки площадки ВЭС напиши на русском
отчёт для руководства (Markdown) для защиты проекта. Разделы строго: «## Резюме», «## Площадка», «## Ветровой ресурс»,
«## Ожидаемая выработка и сравнение с «Нурлы»», «## Сезонность и режим», «## Риски и ограничения», «## Рекомендация».
Первая строка — заголовок «# Оценка площадки ВЭС: <название>». Используй ТОЛЬКО числа из JSON (можно округлять, доли
писать в процентах); не придумывай стоимость, тарифы, окупаемость и другие величины, которых нет в JSON.
Пиши деловым русским языком для руководителей, которые не видят JSON: НИКОГДА не цитируй имена полей и ключей
(никаких «ws100.mean», «cf =», «aep_gwh», «n_turbines», «key:», «in_kazakhstan»), не пиши в стиле «поле = значение»,
не упоминай слово JSON. Каждое число — в предложении с единицей измерения: «средняя скорость ветра на 100 м 7,1 м/с»,
«КИУМ 42,6 %», «годовая выработка 93,3 ГВт·ч». Объясни, что значат цифры для решения о строительстве; честно перечисли
ограничения. Ориентир по стилю и структуре — черновик по шаблону: перепиши его связно и глубже, а не дословно."""

_JSONISH = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\s*[:=]|\b[a-z]+\.[a-z]+\s*=|\bJSON\b|=\s*(?:true|false)\b")


def _looks_like_json_dump(text: str) -> int:
    """Сколько раз текст цитирует поля JSON («cf = 0.42», «ws100.mean =», «in_kazakhstan = true»)."""
    return len(_JSONISH.findall(text))


def management_report(a: dict, forecast_summary: dict | None = None, use_llm: bool = True) -> dict:
    """Отчёт для руководства: {"markdown", "llm_used", "fact_check"}. Шаблон строится всегда; LLM (если есть ключ)
    переписывает его по фактам, затем все числа проверяются verify_narrative — при провале возвращается шаблон."""
    from .agent import llm
    from .agent.tools import verify_narrative
    template = template_report(a, forecast_summary)
    facts = allowed_facts(a, forecast_summary)
    base_check = verify_narrative(template, facts)
    settings = llm.llm_settings() if use_llm else None
    if not settings:
        return {"markdown": template, "llm_used": False,
                "fact_check": {**base_check, "note": "LLM не используется: " + ("отключена" if not use_llm else "нет ключа")}}
    try:
        client = llm.make_client()
        payload = {k: v for k, v in a.items() if k not in ("diurnal",)}
        site = {k: v for k, v in a["site"].items() if k != "transfer"}     # «transfer» LLM путает с передачей в сеть
        site["power_curve_from_nurly"] = bool(a["site"].get("transfer"))
        payload["site"] = site
        if forecast_summary:
            payload["forecast_48h"] = forecast_summary
        messages = [{"role": "system", "content": LLM_SYSTEM},
                    {"role": "user", "content": "JSON оценки:\n" + json.dumps(payload, ensure_ascii=False)
                     + "\n\nЧерновик по шаблону (можно расширить и переписать):\n" + template}]
        check: dict = {}
        for attempt in range(2):                      # одна попытка + один повтор с перечнем неподтверждённых чисел
            resp = client.chat.completions.create(
                model=settings["model"], messages=messages,
                **llm._request_kwargs(settings["model"], settings.get("reasoning_effort")))
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = text.strip("`").removeprefix("markdown").strip()
            if len(text) < 300 or "## Рекомендация" not in text:
                raise ValueError("ответ LLM неполный")
            check = verify_narrative(text, facts)
            jsonish = _looks_like_json_dump(text)
            if check["ok"] and jsonish <= 2:
                return {"markdown": text, "llm_used": True,
                        "fact_check": {**check, "model": settings["model"], "attempts": attempt + 1}}
            notes = []
            if not check["ok"]:
                bad = ", ".join(str(x) for x in check.get("unverified", [])[:10])
                notes.append(f"Проверка чисел не пройдена: этих чисел нет в данных оценки или они получены пересчётом: "
                             f"{bad}. Бери значения ровно как в данных, не складывай и не пересчитывай величины.")
            if jsonish > 2:
                notes.append("Текст цитирует имена полей и пишет в стиле «поле = значение» — так нельзя: перепиши деловым "
                             "русским языком, каждое число в предложении с единицей измерения, без имён полей и слова JSON.")
            check["jsonish"] = jsonish
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": " ".join(notes) + " Перепиши отчёт целиком."}]
        # сверка — для показанного текста (шаблона); отклонённые числа LLM — отдельно в llm_unverified
        why = "неподтверждённые числа" if not check.get("ok") else "стиль (цитирует поля данных)"
        return {"markdown": template, "llm_used": False,
                "fact_check": {**base_check, "llm_unverified": check.get("unverified", []),
                               "note": f"текст LLM отклонён после повтора: {why}, возвращён шаблон"}}
    except Exception as e:  # noqa: BLE001 — LLM опциональна
        log.warning("LLM-отчёт не получен: %s", e)
        return {"markdown": template, "llm_used": False, "fact_check": {**base_check, "note": f"LLM недоступна: {e}"}}
