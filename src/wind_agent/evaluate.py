"""Сверка прогнозов с фактом и файл сдачи.

* `evaluate` — эксперт подкладывает полный файл организаторов (тот же формат, что data/raw/turbine_*.csv, но с февралём)
  и получает MAE/RMSE/смещение/покрытие по турбинам и горизонтам 1–24 / 25–48 ч.
* `build_submission` — единый файл outputs/submission_feb2026.csv: почасовой прогноз по каждой турбине и парку
  в местном времени датасета, отдельно для горизонта 24 ч (выпуск накануне) и 48 ч (выпуск за двое суток).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import RAW_COLUMNS, local_to_utc

log = logging.getLogger(__name__)
FORECASTS_DIR = config.OUTPUTS_DIR / "forecasts"


def load_actual_hourly(path: str | Path) -> pd.Series:
    """Факт в формате датасета (10-мин) → часовое среднее нормированной мощности, индекс UTC."""
    df = pd.read_csv(path)
    df.columns = RAW_COLUMNS[: len(df.columns)]
    df["time_local"] = pd.to_datetime(df["time_local"])
    h = df.set_index("time_local").sort_index()["power"].resample("1h").mean().dropna()
    h.index = local_to_utc(h.index)
    return h


def load_forecasts(forecast_dir: str | Path | None = None) -> pd.DataFrame:
    """Все выпуски outputs/forecasts/<дата>.csv одним кадром (без latest_by_target)."""
    d = Path(forecast_dir or FORECASTS_DIR)
    files = sorted(d.glob("????-??-??.csv"))
    if not files:
        raise FileNotFoundError(f"в {d} нет файлов прогнозов вида YYYY-MM-DD.csv — сначала wind-agent replay")
    fc = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    fc["t_utc"] = pd.to_datetime(fc["target_time_utc"], utc=True)
    return fc


def evaluate(forecasts: pd.DataFrame, actual: dict[str, pd.Series]) -> dict:
    """Метрики прогноза против факта по турбинам и горизонтам (lead_day 1 = 1–24 ч, 2 = 25–48 ч) и в целом.

    actual: {ключ турбины: часовой ряд мощности (UTC)}; парк сравнивается со средним фактом турбин.
    """
    act = {k: v for k, v in actual.items()}
    if len(act) > 1:
        act["farm"] = pd.concat(act.values(), axis=1).mean(axis=1)
    rows = []
    for (turb, lead), g in forecasts.groupby(["turbine", "lead_day"]):
        if turb not in act:
            continue
        y = act[turb].reindex(g["t_utc"]).to_numpy()
        ok = ~np.isnan(y)
        if ok.sum() == 0:
            continue
        p50, p10, p90 = g["p50"].to_numpy()[ok], g["p10"].to_numpy()[ok], g["p90"].to_numpy()[ok]
        yy = y[ok]
        e = p50 - yy
        rows.append({"turbine": turb, "lead_day": int(lead), "n_hours": int(ok.sum()),
                     "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
                     "n_mae_pct": float(100 * np.abs(e).mean()), "coverage_p10_p90_pct": float(100 * ((yy >= p10) & (yy <= p90)).mean()),
                     "mean_actual": float(yy.mean()), "mean_p50": float(p50.mean()),
                     "range": [str(g["t_utc"].min())[:10], str(g["t_utc"].max())[:10]]})
    by = pd.DataFrame(rows)
    overall = {}
    if len(by):
        turb_only = by[by["turbine"] != "farm"]
        w = turb_only["n_hours"]
        overall = {"n_hours": int(w.sum()), "mae": float((turb_only["mae"] * w).sum() / w.sum()),
                   "rmse": float(np.sqrt((turb_only["rmse"] ** 2 * w).sum() / w.sum())),
                   "bias": float((turb_only["bias"] * w).sum() / w.sum()),
                   "coverage_p10_p90_pct": float((turb_only["coverage_p10_p90_pct"] * w).sum() / w.sum())}
    return {"by": rows, "overall": overall}


def format_report(res: dict) -> str:
    L = ["| Ряд | Горизонт | Часов | MAE | RMSE | Смещение | nMAE, % | Покрытие P10–P90, % | Факт ср. | P50 ср. |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["by"]:
        L.append(f"| {r['turbine']} | {'1–24 ч' if r['lead_day'] == 1 else '25–48 ч'} | {r['n_hours']} | {r['mae']:.3f} | {r['rmse']:.3f} | "
                 f"{r['bias']:+.3f} | {r['n_mae_pct']:.1f} | {r['coverage_p10_p90_pct']:.1f} | {r['mean_actual']:.3f} | {r['mean_p50']:.3f} |")
    o = res.get("overall") or {}
    if o:
        L.append(f"| **турбины, все** | 1–48 ч | {o['n_hours']} | **{o['mae']:.3f}** | {o['rmse']:.3f} | {o['bias']:+.3f} | "
                 f"{100 * o['mae']:.1f} | {o['coverage_p10_p90_pct']:.1f} | — | — |")
    return "\n".join(L)


def evaluate_cli(actual_files: list[str], turbines: list[str], forecast_dir: str | None = None) -> int:
    if len(actual_files) != len(turbines):
        log.error("число файлов --actual (%d) не совпадает с числом ключей --turbines (%d)", len(actual_files), len(turbines))
        return 2
    fc = load_forecasts(forecast_dir)
    actual = {t: load_actual_hourly(p) for t, p in zip(turbines, actual_files)}
    for t, s in actual.items():
        log.info("факт %s: %d часов, %s … %s", t, len(s), str(s.index.min())[:16], str(s.index.max())[:16])
    res = evaluate(fc, actual)
    if not res["by"]:
        log.error("нет общих часов между прогнозами (%s … %s) и фактом", str(fc["t_utc"].min())[:10], str(fc["t_utc"].max())[:10])
        return 1
    out_dir = config.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = format_report(res)
    (out_dir / "evaluation.md").write_text("# Сверка прогнозов с фактом\n\n" + md + "\n", encoding="utf-8")
    print(md)
    log.info("результат: outputs/evaluation.json, outputs/evaluation.md")
    return 0


def build_submission(forecast_dir: str | Path | None = None, out_path: Path | None = None) -> Path:
    """Файл сдачи: почасовой прогноз тестового периода по турбинам и парку, горизонты 24 и 48 ч, местное время датасета."""
    fc = load_forecasts(forecast_dir)
    fc = fc.sort_values(["turbine", "t_utc", "lead_day"])
    local = pd.to_datetime(fc["target_time_local"].str.slice(0, 19))
    sub = pd.DataFrame({
        "Статистическое время": local.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "turbine": fc["turbine"].map({"t1": "1", "t2": "2"}).fillna(fc["turbine"]).astype(str),
        "horizon_h": fc["lead_day"].map({1: 24, 2: 48}).astype(int),
        "lead_hours": fc["lead_hours"].astype(int),
        "issue_date": fc["issue_date"],
        "issue_time_local": fc["issue_time_utc"].map(lambda s: str(s)),
        "p10": fc["p10"], "p50": fc["p50"], "p90": fc["p90"],
        "wind_speed_100m_forecast": fc["wind_speed_100m"],
    })
    sub["issue_time_local"] = pd.to_datetime(fc["issue_time_utc"], utc=True).dt.tz_convert("Etc/GMT-5").dt.strftime("%Y-%m-%d %H:%M:%S").values
    out_path = out_path or config.OUTPUTS_DIR / "submission_feb2026.csv"
    sub.to_csv(out_path, index=False)
    log.info("файл сдачи: %s (%d строк, %s … %s)", out_path.relative_to(config.ROOT) if str(out_path).startswith(str(config.ROOT)) else out_path,
             len(sub), sub["Статистическое время"].min(), sub["Статистическое время"].max())
    return out_path
