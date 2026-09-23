"""Сборка данных для 3D-визуализации `viz/index.html` → `viz/data/viz_data.js`.

Что собирается:
  * рельеф — сетка высот из data/terrain/dem_grid.csv (см. scripts/fetch_dem.py);
  * позиции турбин в локальных координатах (метры от центра, x — восток, y — север);
  * роза ветров на 100 м по архиву прогнозов Open-Meteo 2024-02-16..2026-01-31
    (16 румбов × бины скорости 0-3, 3-6, 6-9, 9-12, 12+ м/с, доли %) — за весь период и только февраль;
  * почасовые прогнозы по датам выпуска из outputs/forecasts/*.csv (контракт колонок ниже).
    Если файлов нет (или указан --sample) — генерируется ОБРАЗЕЦ в том же контракте
    моделью models/power_model.joblib по офлайн-кэшу Open-Meteo и кладётся в viz/data/sample_forecasts/.

Результат пишется как `window.VIZ_DATA = {...}`, чтобы страница открывалась по file:// без сервера.
Заодно рисуется viz/wind_rose.png (matplotlib) для README.

Запуск:
  .venv/bin/python scripts/build_viz_data.py            # реальные прогнозы, иначе образец
  .venv/bin/python scripts/build_viz_data.py --sample   # принудительно образец
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

# 48 строк на прогноз — многопоточность OpenMP в sklearn тут только мешает (накладные расходы ×10)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_agent import config  # noqa: E402  (только чтение констант)
# Общие функции сцены живут в пакете (их же зовёт панель оператора для любой площадки и выпуска)
from wind_agent.vizdata import (  # noqa: E402,F401
    bilinear, circ_mean_deg, farm_rows, forecasts_payload, m_per_deg_lon, rose_stats, to_local, _num,
)

TERRAIN_DIR = ROOT / "data" / "terrain"
DEM_CSV = TERRAIN_DIR / "dem_grid.csv"
DEM_META = TERRAIN_DIR / "meta.json"
OSM_JSON = TERRAIN_DIR / "osm_context.json"          # scripts/fetch_osm_context.py
OSM_MATCH_M = 80.0                                   # турбина OSM ближе 80 м к T1/T2 — это она и есть
ROSE_SOURCE = config.CACHE_DIR / "prevruns__t1__2024-02-16__2026-01-31.csv"
FORECASTS_GLOB = str(config.OUTPUTS_DIR / "forecasts" / "*.csv")
VIZ_DIR = ROOT / "viz"
OUT_JS = VIZ_DIR / "data" / "viz_data.js"
SAMPLE_DIR = VIZ_DIR / "data" / "sample_forecasts"
ROSE_PNG = VIZ_DIR / "wind_rose.png"

M_PER_DEG_LAT = 111_132.0
N_SECTORS = 16
SECTOR_NAMES = ["С", "ССВ", "СВ", "ВСВ", "В", "ВЮВ", "ЮВ", "ЮЮВ",
                "Ю", "ЮЮЗ", "ЮЗ", "ЗЮЗ", "З", "ЗСЗ", "СЗ", "ССЗ"]
SPEED_EDGES = [3.0, 6.0, 9.0, 12.0]                    # → бины 0-3, 3-6, 6-9, 9-12, 12+
SPEED_LABELS = ["0–3", "3–6", "6–9", "9–12", "12+"]
ROSE_COLORS = ["#2b4c7e", "#2f7fb5", "#3fb8af", "#f2c14e", "#f46036"]  # те же цвета, что на странице

CONTRACT_COLUMNS = [
    "issue_date", "issue_time_utc", "target_time_utc", "target_time_local", "lead_hours", "lead_day",
    "turbine", "p10", "p50", "p90", "wind_speed_100m", "wind_direction_100m", "wind_speed_10m",
    "wind_gusts_10m", "temperature_2m", "surface_pressure", "weather_source", "model_version",
]
REQUIRED = ["issue_date", "target_time_local", "turbine", "p10", "p50", "p90",
            "wind_speed_100m", "wind_direction_100m", "temperature_2m"]
WEATHER_COLS = ["wind_speed_100m", "wind_direction_100m", "wind_speed_10m", "wind_gusts_10m",
                "temperature_2m", "surface_pressure"]


# ------------------------------------------------------------------ рельеф
def load_terrain() -> tuple[dict, list[dict]]:
    lat0 = sum(v[0] for v in config.TURBINES.values()) / len(config.TURBINES)
    lon0 = sum(v[1] for v in config.TURBINES.values()) / len(config.TURBINES)
    if DEM_CSV.exists() and DEM_META.exists():
        meta = json.loads(DEM_META.read_text())
        g = pd.read_csv(DEM_CSV).sort_values(["j", "i"])
        nx, ny, step = int(meta["nx"]), int(meta["ny"]), float(meta["step_m"])
        elev = g["elevation_m"].to_numpy(float).reshape(ny, nx)
        lat0, lon0 = meta["center"]["lat"], meta["center"]["lon"]
        half = float(meta["half_size_m"])
        source, synthetic = meta.get("source", "Open-Meteo Elevation API"), False
    else:  # без DEM страница всё равно откроется — с плоской подложкой и предупреждением
        print("ВНИМАНИЕ: нет data/terrain/dem_grid.csv — запустите scripts/fetch_dem.py; рельеф будет плоским",
              file=sys.stderr)
        nx = ny = 49
        step, half = 250.0, 6000.0
        elev = np.zeros((ny, nx))
        source, synthetic = "нет DEM (плоская подложка)", True
    x0 = y0 = -half
    terrain = {
        "nx": nx, "ny": ny, "step_m": step, "x0_m": x0, "y0_m": y0, "half_size_m": half,
        "center": {"lat": lat0, "lon": lon0},
        "order": "строки по j (юг → север), внутри строки по i (запад → восток)",
        "elev": [int(round(v)) if float(v).is_integer() else round(float(v), 1) for v in elev.ravel()],
        "min_m": float(elev.min()), "max_m": float(elev.max()), "mean_m": round(float(elev.mean()), 1),
        "source": source, "synthetic": synthetic,
    }
    turbines = []
    for tid, (lat, lon) in config.TURBINES.items():
        x, y = to_local(lat, lon, lat0, lon0)
        turbines.append({"id": tid, "name": tid.upper(), "lat": lat, "lon": lon,
                         "x_m": round(x, 1), "y_m": round(y, 1),
                         "ground_m": round(bilinear(elev, x, y, x0, y0, step), 1)})
    return terrain, turbines


# ------------------------------------------------------------------ контекст OSM
def _assemble_rings(members: list[dict], lat0: float, lon0: float) -> list[list[list[float]]]:
    """Члены мультиполигона (way с geometry) → замкнутые кольца; незамкнутые куски склеиваем по концам."""
    parts = [[to_local(pt["lat"], pt["lon"], lat0, lon0) for pt in m["geometry"]]
             for m in members if m.get("type") == "way" and m.get("geometry")]
    rings = []
    while parts:
        ring = parts.pop(0)
        changed = True
        while changed and ring[0] != ring[-1]:
            changed = False
            for k, q in enumerate(parts):
                if q[0] == ring[-1]:
                    ring += q[1:]
                elif q[-1] == ring[-1]:
                    ring += q[::-1][1:]
                elif q[-1] == ring[0]:
                    ring = q + ring[1:]
                elif q[0] == ring[0]:
                    ring = q[::-1] + ring[1:]
                else:
                    continue
                parts.pop(k)
                changed = True
                break
        if len(ring) >= 3:
            rings.append(ring)
    return rings


def load_osm(lat0: float, lon0: float, turbines: list[dict]) -> dict:
    """Турбины парка, лес и застройка из data/terrain/osm_context.json в локальных координатах рельефа."""
    empty = {"available": False, "turbines": [], "forest": [], "residential": [],
             "note": "нет data/terrain/osm_context.json — запустите scripts/fetch_osm_context.py"}
    if not OSM_JSON.exists():
        print("OSM: нет osm_context.json — слои OSM не добавлены (scripts/fetch_osm_context.py)", file=sys.stderr)
        return empty
    d = json.loads(OSM_JSON.read_text())
    turb, layers = [], {"forest": [], "residential": []}
    r1 = lambda v: round(v, 1)  # noqa: E731
    for el in d.get("elements", []):
        tags = el.get("tags", {})
        if el.get("type") == "node" and tags.get("power") == "generator":
            x, y = to_local(el["lat"], el["lon"], lat0, lon0)
            best = min(turbines, key=lambda t: math.hypot(x - t["x_m"], y - t["y_m"]))
            dist = math.hypot(x - best["x_m"], y - best["y_m"])
            ours = best["id"] if dist <= OSM_MATCH_M else None
            rec = {"osm_id": el["id"], "x_m": r1(x), "y_m": r1(y), "lat": el["lat"], "lon": el["lon"],
                   "ours": ours, "match_m": r1(dist) if ours else None,
                   "model": " ".join(v for v in [tags.get("manufacturer"), tags.get("model")] if v) or None,
                   "power": tags.get("generator:output:electricity")}
            turb.append(rec)
            if ours:  # паспорт нашей турбины из OSM (если заполнен)
                best.update({"osm_id": el["id"], "osm_match_m": r1(dist), "model": rec["model"], "power": rec["power"]})
            continue
        if tags.get("landuse") == "residential":
            kind = "residential"
        elif tags.get("landuse") == "forest" or tags.get("natural") == "wood":
            kind = "forest"
        else:
            continue
        if el.get("type") == "way" and el.get("geometry"):
            outer = [[to_local(pt["lat"], pt["lon"], lat0, lon0) for pt in el["geometry"]]]
            inner = []
        elif el.get("type") == "relation":
            outer = _assemble_rings([m for m in el.get("members", []) if m.get("role", "outer") in ("outer", "")], lat0, lon0)
            inner = _assemble_rings([m for m in el.get("members", []) if m.get("role") == "inner"], lat0, lon0)
        else:
            continue
        rings = [[[round(x), round(y)] for x, y in ring] for ring in outer + inner]
        if not rings:
            continue
        xs = [p[0] for p in rings[0]]
        ys = [p[1] for p in rings[0]]
        layers[kind].append({"osm_id": el["id"], "osm_type": el["type"],
                             "name": tags.get("name:ru") or tags.get("name"),
                             "rings": rings, "n_outer": len(outer),
                             "label_xy": [round(sum(xs) / len(xs)), round(sum(ys) / len(ys))]})
    n_ours = sum(1 for t in turb if t["ours"])
    print(f"OSM: турбин {len(turb)} (из них совпали с T1/T2: {n_ours}), лес {len(layers['forest'])}, "
          f"застройка {len(layers['residential'])}")
    return {"available": True, "source": d.get("source", "OpenStreetMap"), "fetched_at_utc": d.get("fetched_at_utc"),
            "radius_m": d.get("radius_m"), "match_radius_m": OSM_MATCH_M,
            "turbines": turb, "forest": layers["forest"], "residential": layers["residential"],
            "note": "© участники OpenStreetMap (ODbL)"}


# ------------------------------------------------------------------ роза ветров
def _local_month(times_utc: pd.Series) -> np.ndarray:
    try:
        from wind_agent.data import utc_to_local
        return utc_to_local(pd.DatetimeIndex(times_utc)).month
    except Exception:  # запасной вариант: UTC+5
        return (pd.DatetimeIndex(times_utc) + pd.Timedelta(hours=config.UTC_OFFSET_AFTER)).month


def build_wind_rose() -> dict:
    df = pd.read_csv(ROSE_SOURCE, usecols=["time", "wind_speed_100m", "wind_direction_100m"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    ws, wd = df["wind_speed_100m"].to_numpy(float), df["wind_direction_100m"].to_numpy(float)
    feb = _local_month(df["time"]) == 2
    t0, t1 = df["time"].min().strftime("%d.%m.%Y"), df["time"].max().strftime("%d.%m.%Y")
    return {
        "source": f"Open-Meteo Previous Runs (лаг 0, best match), точка T1, {ROSE_SOURCE.name}",
        "height_m": 100,
        "sectors": SECTOR_NAMES, "speed_bins": SPEED_LABELS, "speed_edges": SPEED_EDGES, "colors": ROSE_COLORS,
        "all": rose_stats(ws, wd, f"Весь период {t0}–{t1}"),
        "feb": rose_stats(ws[feb], wd[feb], "Февраль (2024: 16–29.02, 2025: весь месяц)"),
    }


# ------------------------------------------------------------------ прогнозы
def make_sample(dates: list[str]) -> pd.DataFrame:
    """ОБРАЗЕЦ прогнозов в контракте outputs/forecasts: модель + офлайн-кэш Open-Meteo."""
    from wind_agent.features import make_features
    from wind_agent.model import PowerModel
    from wind_agent.weather import WeatherClient

    wx, m = WeatherClient(offline=True), PowerModel.load()
    version = f"power_model.joblib (sample, train_end {m.meta.get('train_end', '?')})"
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for d in dates:
        parts = []
        for t in config.TURBINES:
            f = wx.archived_forecast(t, d)
            f["turbine"] = t
            p = m.predict(make_features(f.set_index("time")))          # p10/p50/p90, индекс = f['time']
            out = pd.DataFrame({
                "issue_date": d,
                "issue_time_utc": pd.to_datetime(f["issue_time_utc"]).dt.strftime("%Y-%m-%d %H:%M"),
                "target_time_utc": pd.to_datetime(f["time"]).dt.strftime("%Y-%m-%d %H:%M"),
                "target_time_local": pd.to_datetime(f["target_local"]).dt.strftime("%Y-%m-%d %H:%M"),
                "lead_hours": f["lead_hours"].astype(int).values,
                "lead_day": f["lead_day"].astype(int).values,
                "turbine": t,
                "p10": p["p10"].values, "p50": p["p50"].values, "p90": p["p90"].values,
            })
            for c in WEATHER_COLS:
                out[c] = f[c].values
            out["weather_source"] = "open-meteo previous-runs (best_match, day1/day2)"
            out["model_version"] = version
            parts.append(out)
        day = pd.concat(parts, ignore_index=True)
        farm = farm_rows(day)
        farm["weather_source"], farm["model_version"] = parts[0]["weather_source"].iloc[0], version
        day = pd.concat([day, farm], ignore_index=True)[CONTRACT_COLUMNS]
        for c in ["p10", "p50", "p90"]:
            day[c] = day[c].round(4)
        day.to_csv(SAMPLE_DIR / f"{d}.csv", index=False)
        frames.append(day)
    print(f"Образец прогнозов: {len(dates)} дат выпуска → {SAMPLE_DIR.relative_to(ROOT)}/")
    return pd.concat(frames, ignore_index=True)


def load_forecast_files() -> pd.DataFrame | None:
    # только файлы выпусков вида 2026-02-10.csv; сводные (latest_by_target.csv и т.п.) пропускаем
    files = sorted(f for f in glob.glob(FORECASTS_GLOB) if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.csv", Path(f).name))
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as exc:  # битый файл не должен ронять сборку
            print(f"  пропускаю {fp}: {exc}", file=sys.stderr)
            continue
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            print(f"  пропускаю {Path(fp).name}: нет колонок {missing}", file=sys.stderr)
            continue
        frames.append(df)
    if not frames:
        return None
    print(f"Прогнозы: {len(frames)} файлов из outputs/forecasts/")
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ PNG розы для README
def draw_rose_png(rose: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = np.deg2rad(np.arange(N_SECTORS) * 22.5)
    width = np.deg2rad(22.5) * 0.9
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(11, 5.6))
    for ax, key in zip(axes, ["all", "feb"]):
        st = rose[key]
        freq = np.array(st["freq_pct"])
        bottom = np.zeros(N_SECTORS)
        for b, (lab, col) in enumerate(zip(SPEED_LABELS, ROSE_COLORS)):
            ax.bar(theta, freq[:, b], width=width, bottom=bottom, color=col, edgecolor="white",
                   linewidth=0.4, label=f"{lab} м/с")
            bottom += freq[:, b]
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(theta[::2])
        ax.set_xticklabels(SECTOR_NAMES[::2])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_rlabel_position(100)
        ax.tick_params(axis="y", labelsize=8, colors="#555")
        ax.set_title(f"{st['label']}\nn = {st['n_hours']} ч, средняя {st['mean_speed_all']:.1f} м/с, "
                     f"преобладает {st['prevailing']['sector']} ({st['prevailing']['pct']:.0f}%)", fontsize=10, pad=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, title="Скорость ветра на 100 м")
    fig.suptitle("Роза ветров на 100 м по прогнозам Open-Meteo, ВЭС (Енбекшиказахский р-н)", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ сборка payload
def load_forecasts(sample: bool = False, sample_dates: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """Прогнозы для сцены: outputs/forecasts/*.csv, иначе (или при sample=True) — ОБРАЗЕЦ моделью по кэшу."""
    fc = None if sample else load_forecast_files()
    if fc is not None:
        return fc, "outputs/forecasts"
    if not sample:
        print("outputs/forecasts/*.csv не найдены — собираю ОБРАЗЕЦ (--sample)")
    dates = sample_dates or [d.strftime("%Y-%m-%d") for d in pd.date_range(config.TEST_ISSUE_START, config.TEST_ISSUE_END)]
    return make_sample(dates), "sample"


def build_payload(forecast: pd.DataFrame, source: str = "outputs/forecasts",
                  rose: dict | None = None) -> dict:
    """Полный payload window.VIZ_DATA для ВЭС «Нурлы»: рельеф, турбины, OSM, роза ветров и выпуски из forecast."""
    terrain, turbines = load_terrain()
    print(f"Рельеф: {terrain['nx']}×{terrain['ny']}, шаг {terrain['step_m']:.0f} м, "
          f"высоты {terrain['min_m']:.0f}–{terrain['max_m']:.0f} м ({terrain['source']})")
    osm = load_osm(terrain["center"]["lat"], terrain["center"]["lon"], turbines)
    if rose is None:
        rose = build_wind_rose()
    print(f"Роза ветров: {rose['all']['n_hours']} ч (весь период), {rose['feb']['n_hours']} ч (февраль); "
          f"преобладает {rose['all']['prevailing']['sector']}")
    issues = forecasts_payload(forecast)
    return {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"),
        "local_tz": config.LOCAL_TZ_NAME,
        "power_units": "доля номинальной мощности (0–1); парк = среднее по T1 и T2",
        "forecast_source": source,
        "terrain": terrain,
        "turbines": turbines,
        "hub_height_m": 100,
        "upwind_cone": {"half_angle_deg": 15, "range_m": 2000},
        "osm": osm,
        "wind_rose": rose,
        "forecasts": issues,
    }


def write_payload(payload: dict, path: Path = OUT_JS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    js = ("// Сгенерировано scripts/build_viz_data.py — не редактировать вручную.\n"
          "window.VIZ_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    path.write_text(js, encoding="utf-8")
    return path


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="Собрать viz/data/viz_data.js для 3D-визуализации")
    ap.add_argument("--sample", action="store_true", help="принудительно сгенерировать образец прогнозов")
    ap.add_argument("--sample-dates", default=None,
                    help="даты выпуска для образца через запятую (по умолчанию весь тестовый период 31.01–27.02.2026)")
    ap.add_argument("--no-png", action="store_true", help="не рисовать viz/wind_rose.png")
    args = ap.parse_args()

    dates = [d.strip() for d in args.sample_dates.split(",") if d.strip()] if args.sample_dates else None
    fc, source = load_forecasts(args.sample, dates)
    payload = build_payload(fc, source)
    issues = payload["forecasts"]
    write_payload(payload, OUT_JS)
    size_kb = OUT_JS.stat().st_size / 1024
    print(f"Готово: {OUT_JS.relative_to(ROOT)} ({size_kb:.0f} КБ), выпусков прогноза: {len(issues)} [{source}]")
    if size_kb > 3 * 1024:
        print("ВНИМАНИЕ: viz_data.js больше 3 МБ", file=sys.stderr)
    if not args.no_png:
        draw_rose_png(payload["wind_rose"], ROSE_PNG)
        print(f"Роза ветров PNG: {ROSE_PNG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
