"""Сетка высот (DEM) вокруг ВЭС для 3D-визуализации.

По умолчанию — ВЭС «Нурлы»: центр между турбинами t1 и t2, квадрат ±6 км, шаг 250 м (49×49 точек).
Для любой другой площадки: --lat/--lon (+ --out-dir, --step, --half-size); панель оператора зовёт
fetch_dem_grid() (реализация — wind_agent.vizdata, чтобы работать и без папки scripts/).
Источник: Open-Meteo Elevation API (Copernicus DEM GLO-90), до 100 точек на запрос, без ключа.

Результат:
  data/terrain/dem_grid.csv  — lat, lon, elevation_m (+ индексы i/j и локальные координаты x/y в метрах)
  data/terrain/meta.json     — центр, шаг, размер, привязка

Скрипт идемпотентный: если файлы уже есть, ничего не качает (перекачать — флаг --force).

Запуск:  .venv/bin/python scripts/fetch_dem.py
         .venv/bin/python scripts/fetch_dem.py --lat 48.0 --lon 68.0 --step 500 --pause 0 --out-dir data/terrain/c48p000_68p000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_agent.config import TURBINES  # noqa: E402  (координаты турбин «Нурлы», только чтение)
from wind_agent.vizdata import (  # noqa: E402,F401  (общая реализация для скрипта и панели)
    ELEVATION_URL, build_grid, fetch_dem_grid, fetch_elevations, m_per_deg_lon,
)

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


def main() -> int:
    ap = argparse.ArgumentParser(description="Скачать сетку высот вокруг ВЭС (Open-Meteo Elevation API)")
    ap.add_argument("--force", action="store_true", help="перекачать, даже если файлы уже есть")
    ap.add_argument("--pause", type=float, default=PAUSE_S, help="пауза между запросами, с (по умолчанию %(default)s)")
    ap.add_argument("--lat", type=float, default=None, help="широта центра (по умолчанию — середина T1/T2 «Нурлы»)")
    ap.add_argument("--lon", type=float, default=None, help="долгота центра")
    ap.add_argument("--out-dir", default=None, help="папка для dem_grid.csv и meta.json (по умолчанию data/terrain)")
    ap.add_argument("--step", type=float, default=STEP_M, help="шаг сетки, м (по умолчанию %(default)s)")
    ap.add_argument("--half-size", type=float, default=HALF_SIZE_M, help="полуразмер квадрата, м (по умолчанию %(default)s)")
    args = ap.parse_args()
    if (args.lat is None) != (args.lon is None):
        ap.error("--lat и --lon задаются вместе")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else OUT_DIR
    grid_csv, meta_json = out_dir / "dem_grid.csv", out_dir / "meta.json"
    if grid_csv.exists() and meta_json.exists() and not args.force:
        print(f"DEM уже есть: {_rel(grid_csv)} — пропускаю (перекачать: --force)")
        return 0

    if args.lat is None:
        lat0 = sum(v[0] for v in TURBINES.values()) / len(TURBINES)
        lon0 = sum(v[1] for v in TURBINES.values()) / len(TURBINES)
        turbines = TURBINES
    else:
        lat0, lon0, turbines = args.lat, args.lon, {"u1": (args.lat, args.lon)}
    n_side = int(round(2 * args.half_size / args.step)) + 1
    print(f"Центр {lat0:.6f}, {lon0:.6f}; сетка {n_side}×{n_side} = {n_side * n_side} точек, шаг {args.step:.0f} м")
    grid = fetch_dem_grid(lat0, lon0, out_dir, half_size_m=args.half_size, step_m=args.step, pause_s=args.pause,
                          progress=lambda m: print(f"  {m}", flush=True), turbines=turbines, force=args.force)
    print(f"Готово: {_rel(grid_csv)}, высоты {grid['elevation_m'].min():.0f}–{grid['elevation_m'].max():.0f} м")
    return 0


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


if __name__ == "__main__":
    raise SystemExit(main())
