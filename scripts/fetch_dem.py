"""Сетка высот (DEM) вокруг ВЭС для 3D-визуализации.

Центр — середина между турбинами t1 и t2, квадрат ±6 км, шаг 250 м (49×49 точек).
Источник: Open-Meteo Elevation API (Copernicus DEM GLO-90), до 100 точек на запрос, без ключа.

Результат:
  data/terrain/dem_grid.csv  — lat, lon, elevation_m (+ индексы i/j и локальные координаты x/y в метрах)
  data/terrain/meta.json     — центр, шаг, размер, привязка

Скрипт идемпотентный: если файлы уже есть, ничего не качает (перекачать — флаг --force).

Запуск:  .venv/bin/python scripts/fetch_dem.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # координаты турбин берём из конфига проекта (только чтение)
    from wind_agent.config import TURBINES
except Exception:  # pragma: no cover — запасной вариант, если пакет недоступен
    TURBINES = {"t1": (43.645150, 78.535604), "t2": (43.643198, 78.538828)}

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
OUT_DIR = ROOT / "data" / "terrain"
GRID_CSV = OUT_DIR / "dem_grid.csv"
META_JSON = OUT_DIR / "meta.json"

HALF_SIZE_M = 6000.0   # полуразмер квадрата, м
STEP_M = 250.0         # шаг сетки, м
BATCH = 100            # лимит точек на запрос Elevation API
# Бесплатный Open-Meteo считает каждую точку отдельным вызовом, лимит ~600 вызовов/мин →
# не больше ~5 запросов по 100 точек в минуту. Пауза 11 с держит нас под лимитом (~4–5 мин на всю сетку).
PAUSE_S = 11.0
PARTIAL_JSON = OUT_DIR / ".dem_partial.json"   # докачка после обрыва
M_PER_DEG_LAT = 111_132.0


def m_per_deg_lon(lat: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat))


def build_grid(lat0: float, lon0: float) -> pd.DataFrame:
    """Регулярная метрическая сетка (x — на восток, y — на север) → lat/lon (равнопромежуточная проекция)."""
    n = int(round(2 * HALF_SIZE_M / STEP_M)) + 1
    rows = []
    for j in range(n):            # j растёт на север
        y = -HALF_SIZE_M + j * STEP_M
        for i in range(n):        # i растёт на восток
            x = -HALF_SIZE_M + i * STEP_M
            rows.append({"j": j, "i": i, "x_m": x, "y_m": y,
                         "lat": round(lat0 + y / M_PER_DEG_LAT, 6),
                         "lon": round(lon0 + x / m_per_deg_lon(lat0), 6)})
    return pd.DataFrame(rows)


def fetch_elevations(lat: list[float], lon: list[float], session: requests.Session) -> list[float]:
    params = {"latitude": ",".join(f"{v:.6f}" for v in lat), "longitude": ",".join(f"{v:.6f}" for v in lon)}
    last = ""
    for attempt in range(6):
        try:
            r = session.get(ELEVATION_URL, params=params, timeout=60)
            if r.status_code == 429:           # минутный лимит — ждём, пока окно сбросится
                last = r.text[:200]
                print(f"  429 (лимит Open-Meteo): {last} — жду 65 с", file=sys.stderr)
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
            print(f"  повтор после ошибки: {exc}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Elevation API недоступен: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Скачать сетку высот вокруг ВЭС (Open-Meteo Elevation API)")
    ap.add_argument("--force", action="store_true", help="перекачать, даже если файлы уже есть")
    ap.add_argument("--pause", type=float, default=PAUSE_S, help="пауза между запросами, с (по умолчанию %(default)s)")
    args = ap.parse_args()

    if GRID_CSV.exists() and META_JSON.exists() and not args.force:
        print(f"DEM уже есть: {GRID_CSV.relative_to(ROOT)} — пропускаю (перекачать: --force)")
        return 0

    lat0 = sum(v[0] for v in TURBINES.values()) / len(TURBINES)
    lon0 = sum(v[1] for v in TURBINES.values()) / len(TURBINES)
    grid = build_grid(lat0, lon0)
    n_side = int(grid["i"].max()) + 1
    print(f"Центр {lat0:.6f}, {lon0:.6f}; сетка {n_side}×{n_side} = {len(grid)} точек, шаг {STEP_M:.0f} м")

    # докачка: уже полученные пачки лежат в .dem_partial.json (ключ — центр и шаг сетки)
    key = f"{lat0:.6f},{lon0:.6f},{STEP_M},{HALF_SIZE_M}"
    done: dict[str, list[float]] = {}
    if PARTIAL_JSON.exists() and not args.force:
        cached = json.loads(PARTIAL_JSON.read_text())
        if cached.get("key") == key:
            done = cached["batches"]
            print(f"  докачка: уже есть {len(done)} пачек")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    elevations: list[float] = []
    with requests.Session() as s:
        s.headers["User-Agent"] = "HackAlem-wind-viz/0.1 (hackathon demo)"
        n_batches = math.ceil(len(grid) / BATCH)
        fetched_any = False
        for b in range(n_batches):
            part = grid.iloc[b * BATCH:(b + 1) * BATCH]
            if str(b) in done:
                elevations += done[str(b)]
                continue
            if fetched_any:
                time.sleep(args.pause)
            vals = fetch_elevations(part["lat"].tolist(), part["lon"].tolist(), s)
            fetched_any = True
            done[str(b)] = vals
            elevations += vals
            PARTIAL_JSON.write_text(json.dumps({"key": key, "batches": done}))
            print(f"  запрос {b + 1}/{n_batches}: {len(part)} точек", flush=True)
    grid["elevation_m"] = elevations

    grid[["lat", "lon", "elevation_m", "i", "j", "x_m", "y_m"]].to_csv(GRID_CSV, index=False)
    meta = {
        "source": "Open-Meteo Elevation API (Copernicus DEM GLO-90)",
        "url": ELEVATION_URL,
        "center": {"lat": round(lat0, 6), "lon": round(lon0, 6)},
        "half_size_m": HALF_SIZE_M,
        "step_m": STEP_M,
        "nx": n_side,
        "ny": n_side,
        "projection": "равнопромежуточная: x = (lon-lon0)·111320·cos(lat0), y = (lat-lat0)·111132",
        "index": "i — на восток (x), j — на север (y); строки CSV упорядочены по j, затем по i",
        "elevation_min_m": float(grid["elevation_m"].min()),
        "elevation_max_m": float(grid["elevation_m"].max()),
        "turbines": {k: {"lat": v[0], "lon": v[1]} for k, v in TURBINES.items()},
        "fetched_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
    }
    META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    PARTIAL_JSON.unlink(missing_ok=True)
    print(f"Готово: {GRID_CSV.relative_to(ROOT)}, высоты {meta['elevation_min_m']:.0f}–{meta['elevation_max_m']:.0f} м")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
