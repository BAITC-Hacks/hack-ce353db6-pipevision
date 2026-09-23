"""Атлас ветрового ресурса Казахстана: климатология ветра на 50 м по NASA POWER (MERRA-2, 2001–2020).

Сетка MERRA-2: 0.5° по широте × 0.625° по долготе. Региональный запрос NASA POWER ограничен 10°×10°,
поэтому bbox Казахстана (широта 40–56, долгота 46–88) покрываем перекрывающимися плитками и убираем дубли.
Ветер на 100 м оцениваем степенным законом (показатель 0.143 — нейтральная стратификация, открытая местность).

Результат — data/atlas/kz_wind_atlas.csv (коммитится: панель на сервере читает только его, без сети)
и data/atlas/atlas_meta.json. Маска: точки внутри полигона границы (data/atlas/kaz.geo.json) плюс точки
ближе 0.35° к границе, чтобы сетка не «обгрызла» край страны (колонка inside_kz).
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

ATLAS_DIR = config.ROOT / "data" / "atlas"
ATLAS_CSV = ATLAS_DIR / "kz_wind_atlas.csv"
ATLAS_META = ATLAS_DIR / "atlas_meta.json"
KAZ_GEOJSON = ATLAS_DIR / "kaz.geo.json"

POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/regional"
BBOX = dict(lat_min=40.0, lat_max=56.0, lon_min=46.0, lon_max=88.0)
# перекрывающиеся плитки ≤ 10° (перекрытие 0.5°, чтобы узлы на границах плиток не потерялись)
LAT_TILES = [(40.0, 49.5), (49.0, 56.0)]
LON_TILES = [(46.0, 55.5), (55.0, 64.5), (64.0, 73.5), (73.0, 82.5), (82.0, 88.0)]
DLAT, DLON = 0.5, 0.625          # шаг сетки MERRA-2
BORDER_BUFFER_DEG = 0.35         # точки ближе этого к границе тоже считаем «своими»
SHEAR_ALPHA = 0.143              # степенной закон: ws(h) = ws50 · (h/50)^α
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
CLASS_EDGES = [(5.5, "низкий"), (6.5, "умеренный"), (7.5, "хороший")]   # по ws100_est; ≥ 7.5 — «отличный»
COLUMNS = (["lat", "lon", "elevation_m", "ws50_ann"] + [f"ws50_m{i:02d}" for i in range(1, 13)]
           + ["ws100_est", "resource_class", "inside_kz"])

_ATLAS: pd.DataFrame | None = None
_POLY: list[tuple[float, float]] | None = None


# ---------------------------------------------------------------- граница и маска
def kz_polygon() -> list[tuple[float, float]]:
    """Внешний контур границы Казахстана: [(lon, lat), ...] (замкнутый — первая точка = последняя)."""
    global _POLY
    if _POLY is None:
        gj = json.loads(KAZ_GEOJSON.read_text(encoding="utf-8"))
        geom = gj["features"][0]["geometry"] if gj.get("type") == "FeatureCollection" else gj.get("geometry", gj)
        ring = geom["coordinates"][0] if geom["type"] == "Polygon" else max(
            (p[0] for p in geom["coordinates"]), key=len)          # MultiPolygon — берём крупнейший контур
        pts = [(float(x), float(y)) for x, y, *_ in ring]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        _POLY = pts
    return _POLY


def _points_in_polygon(lat: np.ndarray, lon: np.ndarray, poly: list[tuple[float, float]]) -> np.ndarray:
    """Ray casting (луч вдоль долготы) для массива точек."""
    x, y = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
        cross = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cross & (x < xint)
    return inside


def _dist_to_border_deg(lat: np.ndarray, lon: np.ndarray, poly: list[tuple[float, float]]) -> np.ndarray:
    """Минимальное расстояние (в градусах, плоско) от точек до отрезков границы."""
    p = np.array(poly, dtype=float)
    a, b = p[:-1], p[1:]                                  # отрезки (lon, lat)
    px, py = np.asarray(lon, dtype=float)[:, None], np.asarray(lat, dtype=float)[:, None]
    dx, dy = (b[:, 0] - a[:, 0])[None, :], (b[:, 1] - a[:, 1])[None, :]
    l2 = np.where(dx ** 2 + dy ** 2 == 0, 1e-12, dx ** 2 + dy ** 2)
    t = np.clip(((px - a[None, :, 0]) * dx + (py - a[None, :, 1]) * dy) / l2, 0, 1)
    cx, cy = a[None, :, 0] + t * dx, a[None, :, 1] + t * dy
    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2).min(axis=1)


def in_kazakhstan(lat: float, lon: float) -> bool:
    """Точка внутри полигона границы Казахстана (без буфера)."""
    return bool(_points_in_polygon(np.array([lat]), np.array([lon]), kz_polygon())[0])


def resource_class(ws100: float) -> str:
    """Класс ветрового ресурса по средней скорости на 100 м: низкий / умеренный / хороший / отличный."""
    if ws100 is None or not np.isfinite(ws100):
        return "нет данных"
    for edge, name in CLASS_EDGES:
        if ws100 < edge:
            return name
    return "отличный"


# ---------------------------------------------------------------- сборка
def _fetch_tile(lat0: float, lat1: float, lon0: float, lon1: float, retries: int = 4) -> list[dict]:
    params = {"parameters": "WS50M", "community": "RE", "format": "JSON",
              "latitude-min": lat0, "latitude-max": lat1, "longitude-min": lon0, "longitude-max": lon1}
    for attempt in range(retries):
        try:
            r = requests.get(POWER_URL, params=params, timeout=60)
            if r.status_code == 429:
                raise RuntimeError("429 Too Many Requests")
            r.raise_for_status()
            return r.json().get("features", [])
        except Exception as e:  # noqa: BLE001 — сеть: повторяем с паузой
            log.warning("NASA POWER плитка %s–%s × %s–%s: %s (попытка %d)", lat0, lat1, lon0, lon1, e, attempt + 1)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"NASA POWER недоступен для плитки {lat0}–{lat1} × {lon0}–{lon1}")


def _rows_from_features(features: list[dict]) -> list[dict]:
    rows = []
    for f in features:
        lon, lat, *rest = f["geometry"]["coordinates"]
        ws = (f.get("properties") or {}).get("parameter", {}).get("WS50M") or {}
        ann = ws.get("ANN")
        if ann is None or ann <= -900:                       # fill_value −999
            continue
        row = {"lat": round(float(lat), 4), "lon": round(float(lon), 4),
               "elevation_m": round(float(rest[0]), 1) if rest and rest[0] is not None and rest[0] > -900 else np.nan,
               "ws50_ann": float(ann)}
        for i, m in enumerate(MONTHS, start=1):
            v = ws.get(m)
            row[f"ws50_m{i:02d}"] = float(v) if v is not None and v > -900 else np.nan
        rows.append(row)
    return rows


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(["lat", "lon"]).sort_values(["lat", "lon"]).reset_index(drop=True)
    df["ws100_est"] = (df["ws50_ann"] * (100 / 50) ** SHEAR_ALPHA).round(2)
    df["resource_class"] = df["ws100_est"].map(resource_class)
    poly = kz_polygon()
    inside = _points_in_polygon(df["lat"].to_numpy(), df["lon"].to_numpy(), poly)
    near = _dist_to_border_deg(df["lat"].to_numpy(), df["lon"].to_numpy(), poly) < BORDER_BUFFER_DEG
    df["inside_kz"] = inside | near
    return df[COLUMNS]


def build_atlas(force: bool = False, progress=None) -> pd.DataFrame:
    """Скачать климатологию NASA POWER по bbox Казахстана и записать ATLAS_CSV + atlas_meta.json.

    Если файл уже есть и не force — просто читает его. progress(i, n, text) — необязательный колбэк.
    Возвращает полный DataFrame (включая точки вне маски, inside_kz=False).
    """
    global _ATLAS
    if ATLAS_CSV.exists() and not force:
        return pd.read_csv(ATLAS_CSV)
    tiles = [(a, b, c, d) for a, b in LAT_TILES for c, d in LON_TILES]
    rows: list[dict] = []
    for i, (a, b, c, d) in enumerate(tiles, start=1):
        if progress:
            progress(i, len(tiles), f"плитка {a}–{b}° с.ш. × {c}–{d}° в.д.")
        rows += _rows_from_features(_fetch_tile(a, b, c, d))
        time.sleep(0.3)
    df = _finalize(pd.DataFrame(rows))
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ATLAS_CSV, index=False)
    kz = df[df["inside_kz"]]
    meta = {
        "source": "NASA POWER Climatology API (MERRA-2), параметр WS50M — средняя скорость ветра на 50 м",
        "api": POWER_URL, "period": "2001–2020", "grid": f"{DLAT}° × {DLON}° (MERRA-2)",
        "extrapolation_100m": f"степенной закон ws50·(100/50)^{SHEAR_ALPHA}",
        "classes_ws100": {"низкий": "<5.5", "умеренный": "5.5–6.5", "хороший": "6.5–7.5", "отличный": "≥7.5"},
        "mask": f"полигон {KAZ_GEOJSON.name} + буфер {BORDER_BUFFER_DEG}°",
        "built_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_points_total": int(len(df)), "n_cells_kz": int(len(kz)),
        "ws100_est_kz": {"min": float(kz["ws100_est"].min()), "mean": round(float(kz["ws100_est"].mean()), 2),
                         "max": float(kz["ws100_est"].max())},
        "class_counts_kz": {k: int(v) for k, v in kz["resource_class"].value_counts().items()},
    }
    ATLAS_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("атлас: %d точек, в маске Казахстана %d → %s", len(df), len(kz), ATLAS_CSV)
    _ATLAS = None
    return df


# ---------------------------------------------------------------- чтение
def load_atlas() -> pd.DataFrame:
    """Ячейки атласа внутри маски Казахстана (кэшируется в модуле)."""
    global _ATLAS
    if _ATLAS is None:
        df = pd.read_csv(ATLAS_CSV)
        df["inside_kz"] = df["inside_kz"].astype(str).str.lower().isin(["true", "1"])
        _ATLAS = df[df["inside_kz"]].reset_index(drop=True)
    return _ATLAS


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def nearest_cell(lat: float, lon: float) -> dict:
    """Ближайшая ячейка атласа: все колонки строки + distance_km."""
    df = load_atlas()
    d = _haversine_km(lat, lon, df["lat"].to_numpy(), df["lon"].to_numpy())
    i = int(np.argmin(d))
    row = {k: (v.item() if hasattr(v, "item") else v) for k, v in df.iloc[i].to_dict().items()}
    row["distance_km"] = round(float(d[i]), 1)
    return row


def grid_cells_geojson() -> dict:
    """FeatureCollection прямоугольников ячеек (0.5° × 0.625°) вокруг узлов атласа — для хороплета на карте.

    id = f"{lat}_{lon}", properties: lat, lon, ws100_est, ws50_ann, resource_class. Координаты [lon, lat].
    """
    df = load_atlas()
    lons = np.unique(df["lon"].to_numpy())
    steps = np.diff(lons)
    dlon = float(np.min(steps[steps > 1e-6])) if len(steps) and (steps > 1e-6).any() else DLON
    hy, hx = DLAT / 2, dlon / 2
    feats = []
    for r in df.itertuples(index=False):
        lat, lon = float(r.lat), float(r.lon)
        ring = [[lon - hx, lat - hy], [lon + hx, lat - hy], [lon + hx, lat + hy], [lon - hx, lat + hy], [lon - hx, lat - hy]]
        feats.append({"type": "Feature", "id": f"{lat}_{lon}",
                      "geometry": {"type": "Polygon", "coordinates": [[[round(x, 4), round(y, 4)] for x, y in ring]]},
                      "properties": {"lat": lat, "lon": lon, "ws100_est": float(r.ws100_est),
                                     "ws50_ann": float(r.ws50_ann), "resource_class": str(r.resource_class)}})
    return {"type": "FeatureCollection", "features": feats}


def top_cells(n: int = 10) -> pd.DataFrame:
    """Лучшие ячейки по ws100_est (для подсказки «где строить»)."""
    return load_atlas().nlargest(n, "ws100_est").reset_index(drop=True)
