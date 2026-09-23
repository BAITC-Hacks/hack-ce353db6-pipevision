"""Данные 3D-сцены `viz/index.html` для любой площадки и любого выпуска прогноза.

API для панели оператора (без streamlit):
  viz_payload(site, forecast, ...) → dict в контракте window.VIZ_DATA;
  viz_html(payload, issue_date)    → самодостаточная страница для st.components.v1.html (srcdoc);
  terrain_ready(site)              → есть ли рельеф площадки на диске (иначе загрузка DEM ~1 мин).

ВЭС «Нурлы»: рельеф, турбины, слои OSM и роза ветров берутся из готового viz/data/viz_data.js
(scripts/build_viz_data.py), а прогнозы заменяются текущим выпуском панели.
Своя площадка: сетка высот Open-Meteo Elevation API 25×25 с шагом 500 м (±6 км) в data/terrain/<site.key>/,
N турбин рядом поперёк преобладающего ветра, роза ветров — по самому прогнозу (48 ч).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import config

VIZ_DIR = config.ROOT / "viz"
VIZ_PAGE = VIZ_DIR / "index.html"
VIZ_DATA_JS = VIZ_DIR / "data" / "viz_data.js"
VIZ_DATA_TAG = '<script src="data/viz_data.js"></script>'
TERRAIN_ROOT = config.ROOT / "data" / "terrain"          # рельеф «Нурлы» лежит прямо здесь, свои площадки — в подпапках

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
ELEVATION_BATCH = 100              # лимит точек на запрос Elevation API
M_PER_DEG_LAT = 111_132.0
SITE_HALF_SIZE_M = 6000.0          # своя площадка: квадрат ±6 км
SITE_STEP_M = 500.0                # … с шагом 500 м → 25×25 = 625 точек, 7 запросов
TURBINE_SPACING_M = 500.0          # шаг турбин в ряду
ROW_MAX = 21                       # турбин в одном ряду (±5 км), дальше — следующий ряд по ветру
ROW_SPACING_M = 1000.0
MAX_TURBINES_DRAWN = 120

N_SECTORS = 16
SECTOR_NAMES = ["С", "ССВ", "СВ", "ВСВ", "В", "ВЮВ", "ЮВ", "ЮЮВ",
                "Ю", "ЮЮЗ", "ЮЗ", "ЗЮЗ", "З", "ЗСЗ", "СЗ", "ССЗ"]
SPEED_EDGES = [3.0, 6.0, 9.0, 12.0]                    # → бины 0-3, 3-6, 6-9, 9-12, 12+
SPEED_LABELS = ["0–3", "3–6", "6–9", "9–12", "12+"]
ROSE_COLORS = ["#2b4c7e", "#2f7fb5", "#3fb8af", "#f2c14e", "#f46036"]
WEATHER_COLS = ["wind_speed_100m", "wind_direction_100m", "wind_speed_10m", "wind_gusts_10m",
                "temperature_2m", "surface_pressure"]
OSM_EMPTY = {"available": False, "turbines": [], "forest": [], "residential": []}

Progress = Callable[[str], None] | None


def _say(progress: Progress, msg: str) -> None:
    if progress is not None:
        try:
            progress(msg)
        except Exception:  # колбэк панели не должен ронять сборку сцены
            pass


# ------------------------------------------------------------------ геометрия
def m_per_deg_lon(lat: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat))


def to_local(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """lat/lon → метры от центра (x — восток, y — север), равнопромежуточная проекция (как в fetch_dem)."""
    return (lon - lon0) * m_per_deg_lon(lat0), (lat - lat0) * M_PER_DEG_LAT


def bilinear(elev: np.ndarray, x: float, y: float, x0: float, y0: float, step: float) -> float:
    ny, nx = elev.shape
    fi = min(max((x - x0) / step, 0), nx - 1.000001)
    fj = min(max((y - y0) / step, 0), ny - 1.000001)
    i, j = int(fi), int(fj)
    di, dj = fi - i, fj - j
    return float(elev[j, i] * (1 - di) * (1 - dj) + elev[j, i + 1] * di * (1 - dj)
                 + elev[j + 1, i] * (1 - di) * dj + elev[j + 1, i + 1] * di * dj)


# ------------------------------------------------------------------ сетка высот (Open-Meteo Elevation API)
def build_grid(lat0: float, lon0: float, half_size_m: float = 6000.0, step_m: float = 250.0) -> pd.DataFrame:
    """Регулярная метрическая сетка (x — на восток, y — на север) → lat/lon (равнопромежуточная проекция)."""
    n = int(round(2 * half_size_m / step_m)) + 1
    rows = []
    for j in range(n):            # j растёт на север
        y = -half_size_m + j * step_m
        for i in range(n):        # i растёт на восток
            x = -half_size_m + i * step_m
            rows.append({"j": j, "i": i, "x_m": x, "y_m": y,
                         "lat": round(lat0 + y / M_PER_DEG_LAT, 6),
                         "lon": round(lon0 + x / m_per_deg_lon(lat0), 6)})
    return pd.DataFrame(rows)


def fetch_elevations(lat: list[float], lon: list[float], session, log: Callable[[str], None] | None = None) -> list[float]:
    """Высоты точек одним запросом (до 100 точек). 429 — ждём сброса минутного окна и повторяем."""
    import requests

    params = {"latitude": ",".join(f"{v:.6f}" for v in lat), "longitude": ",".join(f"{v:.6f}" for v in lon)}
    last = ""
    for attempt in range(6):
        try:
            r = session.get(ELEVATION_URL, params=params, timeout=60)
            if r.status_code == 429:           # минутный лимит Open-Meteo — ждём, пока окно сбросится
                last = r.text[:200]
                if log:
                    log(f"лимит Open-Meteo (429), жду 65 с")
                time.sleep(65)
                continue
            r.raise_for_status()
            elev = r.json()["elevation"]
            if len(elev) != len(lat):
                raise ValueError(f"ожидали {len(lat)} высот, получили {len(elev)}")
            return [float(e) for e in elev]
        except (requests.RequestException, ValueError) as exc:
            last = str(exc)
            if attempt == 5:
                raise
            if log:
                log(f"повтор после ошибки: {exc}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Elevation API недоступен: {last}")


def fetch_dem_grid(lat0: float, lon0: float, out_dir, half_size_m: float = 6000.0, step_m: float = 250.0,
                   pause_s: float = 0.0, progress: Progress = None, turbines: dict | None = None,
                   force: bool = False) -> pd.DataFrame:
    """Скачать сетку высот вокруг (lat0, lon0) в out_dir/dem_grid.csv + meta.json (Copernicus DEM GLO-90).

    Пачки по 100 точек; уже полученные пачки лежат в out_dir/.dem_partial.json (докачка после обрыва).
    pause_s — пауза между запросами (лимит ~600 точек/мин; при 429 ждём сами). progress(msg) — на каждую пачку.
    """
    import requests

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_csv, meta_json, partial = out_dir / "dem_grid.csv", out_dir / "meta.json", out_dir / ".dem_partial.json"
    grid = build_grid(lat0, lon0, half_size_m, step_m)
    n_side = int(grid["i"].max()) + 1
    key = f"{lat0:.6f},{lon0:.6f},{step_m},{half_size_m}"
    done: dict[str, list[float]] = {}
    if partial.exists() and not force:
        try:
            cached = json.loads(partial.read_text())
            if cached.get("key") == key:
                done = cached["batches"]
        except Exception:
            done = {}
    n_batches = math.ceil(len(grid) / ELEVATION_BATCH)
    elevations: list[float] = []
    with requests.Session() as s:
        s.headers["User-Agent"] = "HackAlem-wind-viz/0.1 (hackathon demo)"
        fetched_any = False
        for b in range(n_batches):
            part = grid.iloc[b * ELEVATION_BATCH:(b + 1) * ELEVATION_BATCH]
            if str(b) in done:
                elevations += done[str(b)]
                continue
            if fetched_any and pause_s > 0:
                time.sleep(pause_s)
            vals = fetch_elevations(part["lat"].tolist(), part["lon"].tolist(), s,
                                    log=lambda m: _say(progress, f"Рельеф: {m}"))
            fetched_any = True
            done[str(b)] = vals
            elevations += vals
            partial.write_text(json.dumps({"key": key, "batches": done}))
            _say(progress, f"Рельеф: запрос {b + 1}/{n_batches} ({len(part)} точек)")
    grid["elevation_m"] = elevations
    grid[["lat", "lon", "elevation_m", "i", "j", "x_m", "y_m"]].to_csv(grid_csv, index=False)
    meta = {
        "source": "Open-Meteo Elevation API (Copernicus DEM GLO-90)",
        "url": ELEVATION_URL,
        "center": {"lat": round(lat0, 6), "lon": round(lon0, 6)},
        "half_size_m": float(half_size_m),
        "step_m": float(step_m),
        "nx": n_side,
        "ny": n_side,
        "projection": "равнопромежуточная: x = (lon-lon0)·111320·cos(lat0), y = (lat-lat0)·111132",
        "index": "i — на восток (x), j — на север (y); строки CSV упорядочены по j, затем по i",
        "elevation_min_m": float(grid["elevation_m"].min()),
        "elevation_max_m": float(grid["elevation_m"].max()),
        "turbines": {k: {"lat": v[0], "lon": v[1]} for k, v in (turbines or {}).items()},
        "fetched_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    partial.unlink(missing_ok=True)
    return grid


# ------------------------------------------------------------------ рельеф
def terrain_dir_for(site) -> Path:
    return TERRAIN_ROOT if getattr(site, "key", "") == config.NURLY.key else TERRAIN_ROOT / site.key


def terrain_ready(site) -> bool:
    """Есть ли рельеф площадки на диске (для «Нурлы» достаточно готового viz_data.js)."""
    if getattr(site, "key", "") == config.NURLY.key and VIZ_DATA_JS.exists():
        return True
    d = terrain_dir_for(site)
    return (d / "dem_grid.csv").exists() and (d / "meta.json").exists()


def _site_center(site) -> tuple[float, float]:
    pts = list(site.turbines.values())
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def load_terrain_grid(terrain_dir, lat0: float, lon0: float, half: float = 6000.0, step: float = 250.0) -> tuple[dict, np.ndarray]:
    """dem_grid.csv + meta.json → блок terrain контракта VIZ_DATA и матрица высот (ny, nx).
    Без файлов — плоская подложка с пометкой synthetic (страница всё равно откроется)."""
    terrain_dir = Path(terrain_dir)
    dem_csv, dem_meta = terrain_dir / "dem_grid.csv", terrain_dir / "meta.json"
    if dem_csv.exists() and dem_meta.exists():
        meta = json.loads(dem_meta.read_text())
        g = pd.read_csv(dem_csv).sort_values(["j", "i"])
        nx, ny, step = int(meta["nx"]), int(meta["ny"]), float(meta["step_m"])
        elev = g["elevation_m"].to_numpy(float).reshape(ny, nx)
        lat0, lon0 = meta["center"]["lat"], meta["center"]["lon"]
        half = float(meta["half_size_m"])
        source, synthetic = meta.get("source", "Open-Meteo Elevation API"), False
    else:
        nx = ny = int(round(2 * half / step)) + 1
        elev = np.zeros((ny, nx))
        source, synthetic = "нет DEM (плоская подложка)", True
    terrain = {
        "nx": nx, "ny": ny, "step_m": step, "x0_m": -half, "y0_m": -half, "half_size_m": half,
        "center": {"lat": lat0, "lon": lon0},
        "order": "строки по j (юг → север), внутри строки по i (запад → восток)",
        "elev": [int(round(v)) if float(v).is_integer() else round(float(v), 1) for v in elev.ravel()],
        "min_m": float(elev.min()), "max_m": float(elev.max()), "mean_m": round(float(elev.mean()), 1),
        "source": source, "synthetic": synthetic,
    }
    return terrain, elev


def layout_turbines(n: int, lat0: float, lon0: float, prevailing_deg: float | None, elev: np.ndarray,
                    terrain: dict, first_id: str = "u1") -> list[dict]:
    """N турбин рядами поперёк преобладающего ветра (без розы — ряд запад–восток), шаг 500 м, центр — площадка."""
    n = max(1, min(int(n or 1), MAX_TURBINES_DRAWN))
    wd = 0.0 if prevailing_deg is None else float(prevailing_deg)      # откуда дует; ряд — поперёк (wd + 90°)
    row_b, wind_b = math.radians(wd + 90.0), math.radians(wd)
    n_rows = math.ceil(n / ROW_MAX)
    prefix = first_id.rstrip("0123456789") or "u"
    out, k = [], 0
    for r in range(n_rows):
        in_row = min(ROW_MAX, n - k)
        off = (r - (n_rows - 1) / 2) * ROW_SPACING_M                     # ряды разнесены вдоль ветра
        for c in range(in_row):
            d = (c - (in_row - 1) / 2) * TURBINE_SPACING_M
            x = d * math.sin(row_b) + off * math.sin(wind_b)
            y = d * math.cos(row_b) + off * math.cos(wind_b)
            lat = lat0 + y / M_PER_DEG_LAT
            lon = lon0 + x / m_per_deg_lon(lat0)
            tid = first_id if k == 0 else f"{prefix}{k + 1}"
            out.append({"id": tid, "name": tid.upper(), "lat": round(lat, 6), "lon": round(lon, 6),
                        "x_m": round(x, 1), "y_m": round(y, 1),
                        "ground_m": round(bilinear(elev, x, y, terrain["x0_m"], terrain["y0_m"], terrain["step_m"]), 1)})
            k += 1
    return out


# ------------------------------------------------------------------ роза ветров
def rose_stats(ws: np.ndarray, wd: np.ndarray, label: str) -> dict:
    ok = np.isfinite(ws) & np.isfinite(wd)
    ws, wd = ws[ok], np.mod(wd[ok], 360.0)
    sector = (np.floor((wd + 11.25) / 22.5).astype(int)) % N_SECTORS      # 0 = С (348.75..11.25°)
    sbin = np.digitize(ws, SPEED_EDGES)                                     # 0..4
    counts = np.zeros((N_SECTORS, len(SPEED_LABELS)), dtype=int)
    np.add.at(counts, (sector, sbin), 1)
    n = int(counts.sum())
    pct = counts / max(n, 1) * 100.0
    mean_speed = [round(float(ws[sector == k].mean()), 2) if (sector == k).any() else 0.0 for k in range(N_SECTORS)]
    sector_pct = pct.sum(axis=1)
    k_max = int(sector_pct.argmax())
    return {
        "label": label, "n_hours": n,
        "freq_pct": [[round(float(v), 3) for v in row] for row in pct],
        "sector_pct": [round(float(v), 2) for v in sector_pct],
        "mean_speed": mean_speed,
        "mean_speed_all": round(float(ws.mean()), 2) if n else 0.0,
        "calm_pct": round(float((ws < 1.0).mean() * 100), 2) if n else 0.0,
        "prevailing": {"sector": SECTOR_NAMES[k_max], "deg": k_max * 22.5, "pct": round(float(sector_pct[k_max]), 1)},
        "speed_bin_pct": [round(float(v), 2) for v in pct.sum(axis=0)],
    }


def rose_from_forecast(forecast: pd.DataFrame) -> dict:
    """Роза ветров по самому прогнозу (48 ч ряда парка или первой турбины) — для площадки без архива."""
    base = {"source": "прогноз Open-Meteo текущего выпуска", "height_m": 100, "sectors": SECTOR_NAMES,
            "speed_bins": SPEED_LABELS, "speed_edges": SPEED_EDGES, "colors": ROSE_COLORS}
    if forecast is None or forecast.empty or "wind_speed_100m" not in forecast or "wind_direction_100m" not in forecast:
        return {**base, "all": rose_stats(np.array([]), np.array([]), "нет данных")}
    df = forecast.copy()
    df["turbine"] = df["turbine"].astype(str).str.lower()
    ids = [t for t in df["turbine"].unique() if t != "farm"]
    g = df[df["turbine"] == ids[0]] if ids else df[df["turbine"] == "farm"]
    if g.empty:
        g = df
    g = g.drop_duplicates("target_time_utc" if "target_time_utc" in g else "target_time_local").head(config.HORIZON_HOURS)
    st = rose_stats(g["wind_speed_100m"].to_numpy(float), g["wind_direction_100m"].to_numpy(float),
                    f"Прогноз {len(g)} ч (выпуск {str(g['issue_date'].iloc[0])[:10] if 'issue_date' in g and len(g) else '—'})")
    return {**base, "all": st}


# ------------------------------------------------------------------ прогнозы
def circ_mean_deg(a: pd.Series) -> float:
    r = np.deg2rad(a.astype(float))
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def _turbine_ids(df: pd.DataFrame) -> list[str]:
    """Ряды турбин в прогнозе (всё, кроме farm): t1/t2 у «Нурлы», u1 у своей площадки."""
    ids = [t for t in pd.unique(df["turbine"].astype(str).str.lower()) if t != "farm"]
    return sorted(ids, key=lambda t: (len(t), t))


def farm_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Строки парка: берём turbine == 'farm', если они есть, иначе среднее по турбинам (мощность в долях номинала)."""
    if (df["turbine"] == "farm").any():
        return df[df["turbine"] == "farm"].copy()
    turb = df[df["turbine"].isin(_turbine_ids(df))]
    agg = {c: "mean" for c in ["p10", "p50", "p90"] + [w for w in WEATHER_COLS if w in turb and w != "wind_direction_100m"]}
    keys = [c for c in ["issue_date", "issue_time_utc", "target_time_utc", "target_time_local", "lead_hours", "lead_day"]
            if c in turb]
    out = turb.groupby(keys, as_index=False).agg(agg)
    wd = turb.groupby(keys)["wind_direction_100m"].apply(circ_mean_deg).reset_index(drop=True)
    out["wind_direction_100m"] = wd.values
    out["turbine"] = "farm"
    return out


def _num(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) else round(v, nd)


def forecasts_payload(df: pd.DataFrame) -> list[dict]:
    """Прогноз в контракте agent/tools.FORECAST_COLUMNS (один или несколько выпусков) → список выпусков сцены."""
    if df is None or df.empty:
        return []
    df = df.copy()
    # берём «настенное» время как есть: и «2026-02-11 00:00», и «2026-02-11T00:00:00+05:00» → «2026-02-11 00:00»
    df["issue_date"] = df["issue_date"].astype(str).str.slice(0, 10)
    df["target_time_local"] = df["target_time_local"].astype(str).str.slice(0, 16).str.replace("T", " ")
    if "target_time_utc" in df:
        df["target_time_utc"] = df["target_time_utc"].astype(str).str.slice(0, 16).str.replace("T", " ")
    df["turbine"] = df["turbine"].astype(str).str.lower()
    issues = []
    for d, g in df.groupby("issue_date", sort=True):
        farm = farm_rows(g).sort_values("target_time_local").drop_duplicates("target_time_local")
        ids = _turbine_ids(g)
        per_t = {t: g[g["turbine"] == t].drop_duplicates("target_time_local").set_index("target_time_local") for t in ids}
        hours = []
        for _, r in farm.iterrows():
            tl = r["target_time_local"]
            rec = {
                "target_local": tl,
                "target_utc": str(r.get("target_time_utc", "")) if pd.notna(r.get("target_time_utc", None)) else None,
                "lead_hours": int(r["lead_hours"]) if "lead_hours" in r and pd.notna(r["lead_hours"]) else None,
                "lead_day": int(r["lead_day"]) if "lead_day" in r and pd.notna(r["lead_day"]) else None,
                "wind_speed_100m": _num(r.get("wind_speed_100m"), 2),
                "wind_direction_100m": _num(r.get("wind_direction_100m"), 1),
                "wind_speed_10m": _num(r.get("wind_speed_10m"), 2),
                "wind_gusts_10m": _num(r.get("wind_gusts_10m"), 2),
                "temperature_2m": _num(r.get("temperature_2m"), 2),
                "p50": _num(r["p50"], 4), "p10": _num(r["p10"], 4), "p90": _num(r["p90"], 4),
            }
            for t, tg in per_t.items():
                rec[f"p50_{t}"] = _num(tg.at[tl, "p50"], 4) if tl in tg.index else None
            hours.append(rec)
        # погода парка могла отсутствовать в строках farm — добираем из первой турбины
        t1 = per_t.get(ids[0]) if ids else None
        for rec in hours:
            for c in ["wind_speed_100m", "wind_direction_100m", "temperature_2m", "wind_gusts_10m", "wind_speed_10m"]:
                if rec[c] is None and t1 is not None and rec["target_local"] in t1.index and c in t1:
                    rec[c] = _num(t1.at[rec["target_local"], c], 2)
        first = g.iloc[0]
        issues.append({
            "issue_date": d,
            "issue_time_utc": str(first.get("issue_time_utc", "")),
            "weather_source": str(first.get("weather_source", "")),
            "model_version": str(first.get("model_version", "")),
            "n_hours": len(hours),
            "hours": hours[:config.HORIZON_HOURS],
        })
    return issues


# ------------------------------------------------------------------ payload
_BASE_CACHE: dict[str, tuple[float, dict]] = {}


def load_base_payload(path: Path = VIZ_DATA_JS) -> dict | None:
    """Готовый viz/data/viz_data.js («Нурлы») → dict (кэш по mtime файла)."""
    path = Path(path)
    if not path.exists():
        return None
    mt = path.stat().st_mtime
    hit = _BASE_CACHE.get(str(path))
    if hit and hit[0] == mt:
        return hit[1]
    txt = path.read_text(encoding="utf-8")
    k = txt.find("window.VIZ_DATA")
    if k < 0:
        return None
    body = txt[txt.index("=", k) + 1:].strip()
    if body.endswith(";"):
        body = body[:-1]
    data = json.loads(body)
    _BASE_CACHE[str(path)] = (mt, data)
    return data


def viz_payload(site, forecast: pd.DataFrame, terrain_dir=None, fetch: bool = True, progress: Progress = None,
                rose: dict | None = None) -> dict:
    """Данные сцены (контракт window.VIZ_DATA) для площадки site и прогноза forecast (FORECAST_COLUMNS).

    «Нурлы» — рельеф/турбины/OSM/роза из viz/data/viz_data.js, прогнозы — из forecast.
    Своя площадка — DEM из data/terrain/<site.key>/ (нет и fetch=True — скачать ~7 запросов, иначе плоская подложка).
    """
    issues = forecasts_payload(forecast)
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")
    is_nurly = getattr(site, "key", "") == config.NURLY.key
    if is_nurly and terrain_dir is None:
        base = load_base_payload()
        if base is not None:
            _say(progress, "Сцена: рельеф и турбины ВЭС «Нурлы» из viz/data/viz_data.js")
            out = dict(base)
            out.update({"generated_at_utc": now, "forecast_source": "panel", "forecasts": issues,
                        "site_name": site.name, "site_key": site.key,
                        "turbine_series": [t["id"] for t in base.get("turbines", [])]})
            if rose is not None:
                out["wind_rose"] = rose
            return out

    lat0, lon0 = _site_center(site)
    tdir = Path(terrain_dir) if terrain_dir is not None else terrain_dir_for(site)
    ready = (tdir / "dem_grid.csv").exists() and (tdir / "meta.json").exists()
    if not ready and fetch:
        _say(progress, f"Рельеф: загрузка сетки высот 25×25 вокруг {lat0:.3f}, {lon0:.3f}")
        try:
            fetch_dem_grid(lat0, lon0, tdir, half_size_m=SITE_HALF_SIZE_M, step_m=SITE_STEP_M,
                           progress=progress, turbines=site.turbines)
        except Exception as exc:  # без сети сцена всё равно строится — на плоской подложке
            _say(progress, f"Рельеф недоступен ({exc}); сцена на плоской подложке")
    terrain, elev = load_terrain_grid(tdir, lat0, lon0, SITE_HALF_SIZE_M, SITE_STEP_M)
    lat0, lon0 = terrain["center"]["lat"], terrain["center"]["lon"]

    if rose is None:
        rose = rose_from_forecast(forecast)
    st_all = (rose or {}).get("all") or {}
    prevailing = st_all.get("prevailing", {}).get("deg") if st_all.get("n_hours") else None
    if is_nurly:  # «Нурлы» без viz_data.js: реальные координаты турбин
        turbines = []
        for tid, (lat, lon) in site.turbines.items():
            x, y = to_local(lat, lon, lat0, lon0)
            turbines.append({"id": tid, "name": tid.upper(), "lat": lat, "lon": lon, "x_m": round(x, 1),
                             "y_m": round(y, 1),
                             "ground_m": round(bilinear(elev, x, y, terrain["x0_m"], terrain["y0_m"], terrain["step_m"]), 1)})
    else:
        first = next(iter(site.turbines), "u1")
        turbines = layout_turbines(site.n_turbines or 1, lat0, lon0, prevailing, elev, terrain, first_id=first)
    _say(progress, f"Сцена: рельеф {terrain['nx']}×{terrain['ny']}, турбин {len(turbines)}")
    rated = f", {site.n_turbines} × {site.rated_mw:g} МВт" if site.rated_mw and site.n_turbines else ""
    return {
        "generated_at_utc": now,
        "local_tz": config.LOCAL_TZ_NAME,
        "power_units": ("доля номинальной мощности (0–1); парк = среднее по турбинам" if is_nurly else
                        f"доля номинальной мощности (0–1); одна расчётная турбина на всю площадку{rated}"),
        "forecast_source": "panel",
        "site_name": site.name,
        "site_key": site.key,
        "turbine_series": [t for t in site.turbines],
        "terrain": terrain,
        "turbines": turbines,
        "hub_height_m": 100,
        "upwind_cone": {"half_angle_deg": 15, "range_m": 2000},
        "osm": dict(OSM_EMPTY),
        "wind_rose": rose,
        "forecasts": issues,
    }


def viz_html(payload: dict, issue_date: str | None = None) -> str:
    """viz/index.html с инлайн-данными (srcdoc-iframe не видит относительных путей) и датой выпуска панели
    в window.VIZ_DEFAULT_ISSUE. Основной модуль стартует, когда iframe стал видимым (скрытые вкладки Streamlit
    дают нулевой размер); если three.js не загрузился с CDN — сообщение на месте сцены."""
    htm = VIZ_PAGE.read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    data = data.replace("</", "<\\/")
    inline = (f"<script>window.VIZ_DEFAULT_ISSUE = {json.dumps(str(issue_date) if issue_date is not None else None)};</script>\n"
              f"<script>\nwindow.VIZ_DATA = {data};\n</script>")
    if VIZ_DATA_TAG in htm:
        htm = htm.replace(VIZ_DATA_TAG, inline, 1)
    else:
        htm = htm.replace("<head>", "<head>\n" + inline, 1)
    if '<script type="module">' in htm and "</body>" in htm:
        htm = htm.replace('<script type="module">', '<script type="text/x-wa-deferred" id="wa-viz-main">', 1)
        loader = """<script>
(function () {
  var orig = window.__showErr;
  window.__showErr = function (msg) { if (!window.__waStarted) return; if (orig) orig(msg); };
  function start() {
    if (window.__waStarted || window.innerWidth < 50 || window.innerHeight < 50) return;
    window.__waStarted = true;
    var src = document.getElementById('wa-viz-main'), s = document.createElement('script');
    s.type = 'module'; s.textContent = src.textContent; document.body.appendChild(s);
    setTimeout(function () { if (!window.__vizStarted && orig) orig('three.js не загрузился с CDN (cdn.jsdelivr.net).'); }, 8000);
  }
  window.addEventListener('resize', start);
  var iv = setInterval(function () { start(); if (window.__waStarted) clearInterval(iv); }, 250);
  start();
})();
</script>
"""
        htm = htm.replace("</body>", loader + "</body>", 1)
    return htm


def write_data_js(payload: dict, path) -> Path:
    """payload → файл `window.VIZ_DATA = {...};` (для открытия viz/index.html по file://)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    js = ("// Сгенерировано wind_agent.vizdata — не редактировать вручную.\n"
          "window.VIZ_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    path.write_text(js, encoding="utf-8")
    return path
