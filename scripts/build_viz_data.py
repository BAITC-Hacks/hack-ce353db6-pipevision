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

TERRAIN_DIR = ROOT / "data" / "terrain"
DEM_CSV = TERRAIN_DIR / "dem_grid.csv"
DEM_META = TERRAIN_DIR / "meta.json"
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


# ------------------------------------------------------------------ геометрия
def m_per_deg_lon(lat: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat))


def to_local(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """lat/lon → метры от центра (x — восток, y — север), та же проекция, что в fetch_dem.py."""
    return (lon - lon0) * m_per_deg_lon(lat0), (lat - lat0) * M_PER_DEG_LAT


def bilinear(elev: np.ndarray, x: float, y: float, x0: float, y0: float, step: float) -> float:
    ny, nx = elev.shape
    fi = min(max((x - x0) / step, 0), nx - 1.000001)
    fj = min(max((y - y0) / step, 0), ny - 1.000001)
    i, j = int(fi), int(fj)
    di, dj = fi - i, fj - j
    return float(elev[j, i] * (1 - di) * (1 - dj) + elev[j, i + 1] * di * (1 - dj)
                 + elev[j + 1, i] * (1 - di) * dj + elev[j + 1, i + 1] * di * dj)


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


# ------------------------------------------------------------------ роза ветров
def _local_month(times_utc: pd.Series) -> np.ndarray:
    try:
        from wind_agent.data import utc_to_local
        return utc_to_local(pd.DatetimeIndex(times_utc)).month
    except Exception:  # запасной вариант: UTC+5
        return (pd.DatetimeIndex(times_utc) + pd.Timedelta(hours=config.UTC_OFFSET_AFTER)).month


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
        "mean_speed_all": round(float(ws.mean()), 2),
        "calm_pct": round(float((ws < 1.0).mean() * 100), 2),
        "prevailing": {"sector": SECTOR_NAMES[k_max], "deg": k_max * 22.5, "pct": round(float(sector_pct[k_max]), 1)},
        "speed_bin_pct": [round(float(v), 2) for v in pct.sum(axis=0)],
    }


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
def circ_mean_deg(a: pd.Series) -> float:
    r = np.deg2rad(a.astype(float))
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def farm_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Строки парка: берём turbine == 'farm', если они есть, иначе среднее по t1/t2 (мощность в долях номинала)."""
    if (df["turbine"] == "farm").any():
        return df[df["turbine"] == "farm"].copy()
    turb = df[df["turbine"].isin(list(config.TURBINES))]
    agg = {c: "mean" for c in ["p10", "p50", "p90"] + [w for w in WEATHER_COLS if w in turb and w != "wind_direction_100m"]}
    keys = [c for c in ["issue_date", "issue_time_utc", "target_time_utc", "target_time_local", "lead_hours", "lead_day"]
            if c in turb]
    out = turb.groupby(keys, as_index=False).agg(agg)
    wd = turb.groupby(keys)["wind_direction_100m"].apply(circ_mean_deg).reset_index(drop=True)
    out["wind_direction_100m"] = wd.values
    out["turbine"] = "farm"
    return out


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


def _num(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) else round(v, nd)


def forecasts_payload(df: pd.DataFrame) -> list[dict]:
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
        per_t = {t: g[g["turbine"] == t].drop_duplicates("target_time_local").set_index("target_time_local")
                 for t in config.TURBINES}
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
        # погода парка могла отсутствовать в строках farm — добираем из t1
        t1 = per_t.get("t1")
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


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description="Собрать viz/data/viz_data.js для 3D-визуализации")
    ap.add_argument("--sample", action="store_true", help="принудительно сгенерировать образец прогнозов")
    ap.add_argument("--sample-dates", default=None,
                    help="даты выпуска для образца через запятую (по умолчанию весь тестовый период 31.01–27.02.2026)")
    ap.add_argument("--no-png", action="store_true", help="не рисовать viz/wind_rose.png")
    args = ap.parse_args()

    terrain, turbines = load_terrain()
    print(f"Рельеф: {terrain['nx']}×{terrain['ny']}, шаг {terrain['step_m']:.0f} м, "
          f"высоты {terrain['min_m']:.0f}–{terrain['max_m']:.0f} м ({terrain['source']})")
    rose = build_wind_rose()
    print(f"Роза ветров: {rose['all']['n_hours']} ч (весь период), {rose['feb']['n_hours']} ч (февраль); "
          f"преобладает {rose['all']['prevailing']['sector']}")

    fc = None if args.sample else load_forecast_files()
    source = "outputs/forecasts"
    if fc is None:
        if not args.sample:
            print("outputs/forecasts/*.csv не найдены — собираю ОБРАЗЕЦ (--sample)")
        if args.sample_dates:
            dates = [d.strip() for d in args.sample_dates.split(",") if d.strip()]
        else:
            dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(config.TEST_ISSUE_START, config.TEST_ISSUE_END)]
        fc = make_sample(dates)
        source = "sample"
    issues = forecasts_payload(fc)

    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"),
        "local_tz": config.LOCAL_TZ_NAME,
        "power_units": "доля номинальной мощности (0–1); парк = среднее по T1 и T2",
        "forecast_source": source,
        "terrain": terrain,
        "turbines": turbines,
        "hub_height_m": 100,
        "wind_rose": rose,
        "forecasts": issues,
    }
    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    js = ("// Сгенерировано scripts/build_viz_data.py — не редактировать вручную.\n"
          "window.VIZ_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    OUT_JS.write_text(js, encoding="utf-8")
    size_kb = OUT_JS.stat().st_size / 1024
    print(f"Готово: {OUT_JS.relative_to(ROOT)} ({size_kb:.0f} КБ), выпусков прогноза: {len(issues)} [{source}]")
    if size_kb > 3 * 1024:
        print("ВНИМАНИЕ: viz_data.js больше 3 МБ", file=sys.stderr)
    if not args.no_png:
        draw_rose_png(rose, ROSE_PNG)
        print(f"Роза ветров PNG: {ROSE_PNG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
