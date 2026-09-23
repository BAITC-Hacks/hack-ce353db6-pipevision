"""Инструменты агента: детерминированные функции полного цикла прогноза.

fetch_weather → prepare_data → run_model → analyze_forecast → write_report / save_forecast.

Инструменты не знают, кто их вызывает — детерминированный оркестратор или LLM через tool calling.
Таблицы (DataFrame) остаются в памяти процесса, наружу (в LLM и в журнал) уходят компактные сводки.
"""
from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..data import load_hourly, utc_to_local
from ..features import make_features
from ..model import PowerModel

log = logging.getLogger(__name__)

SOURCE_ARCHIVE = "open-meteo/previous-runs"
SOURCE_LIVE = "open-meteo/forecast"

# Контракт файла прогноза outputs/forecasts/<issue_date>.csv — порядок колонок фиксирован
FORECAST_COLUMNS = [
    "issue_date", "issue_time_utc", "target_time_utc", "target_time_local", "lead_hours", "lead_day", "turbine",
    "p10", "p50", "p90", "wind_speed_100m", "wind_direction_100m", "wind_speed_10m", "wind_gusts_10m",
    "temperature_2m", "surface_pressure", "weather_source", "model_version",
]
WEATHER_OUT = ["wind_speed_100m", "wind_direction_100m", "wind_speed_10m", "wind_gusts_10m", "temperature_2m", "surface_pressure"]
SERIES = ["t1", "t2", "farm"]

# Пороги проверок
SPEED_VARS = ["wind_speed_10m", "wind_speed_80m", "wind_speed_100m", "wind_speed_120m", "wind_gusts_10m"]
SPEED_RANGE = (0.0, 60.0)          # м/с
TEMP_RANGE = (-50.0, 50.0)         # °C
PRESSURE_RANGE = (700.0, 1100.0)   # гПа, площадка ~555 м над уровнем моря
RAMP_THRESHOLD = 0.4               # часовой скачок P50 парка, доля номинала
REVISION_MAE_THRESHOLD = 0.15      # «существенная ревизия» к предыдущему выпуску
WIDE_BAND_THRESHOLD = 0.6          # средняя ширина P90−P10 — низкая уверенность
CALM_LEVEL = 0.05                  # P50 ≤ 5 % номинала — практически без генерации
CALM_MIN_HOURS = 6                 # окно штиля от 6 ч — кандидат на окно для ТО
HIGH_LEVEL = 0.9                   # P50 ≥ 90 % номинала


# ---------------------------------------------------------------- вспомогательное
@lru_cache(maxsize=1)
def load_model() -> PowerModel:
    """Модель загружается один раз на процесс."""
    return PowerModel.load()


def model_version(model: PowerModel | None = None) -> str:
    """Строка версии модели: power_model_v1@<дата конца обучения> (из PowerModel.meta)."""
    model = model or load_model()
    train_end = str(model.meta.get("train_end", ""))[:10]
    return f"power_model_v1@{train_end}" if train_end else "power_model_v1"


def iso_utc(ts) -> pd.Series:
    """Метки UTC → ISO 8601 с явным смещением: 2026-02-01T00:00:00+00:00."""
    idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    return pd.Series(idx.strftime("%Y-%m-%dT%H:%M:%S") + "+00:00")


def iso_local(ts) -> pd.Series:
    """Метки UTC → местное время ISO 8601 со смещением правил датасета: 2026-02-01T05:00:00+05:00."""
    utc = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    local = utc_to_local(utc)
    offset_h = ((local - utc.tz_localize(None)) / pd.Timedelta(hours=1)).astype(int)
    suffix = pd.Index([f"+{h:02d}:00" for h in offset_h])
    return pd.Series(local.strftime("%Y-%m-%dT%H:%M:%S") + suffix)


def input_hash(frames: dict[str, pd.DataFrame]) -> str:
    """Хэш входа (время + погодные переменные по всем турбинам) — чтобы понять, обновились ли данные."""
    h = hashlib.sha256()
    for t in sorted(frames):
        df = frames[t]
        h.update(t.encode())
        h.update(iso_utc(df["time"]).str.cat(sep="|").encode())
        h.update(np.round(df[config.WEATHER_VARS].to_numpy(dtype=float), 3).tobytes())
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def load_history() -> dict:
    """Климатология по истории SCADA: средняя нормированная мощность по месяцам (местное время).

    Возвращает {'by_month': {t1: {1: .., 2: ..}, t2: {...}, farm: {...}}, 'range': [..]}.
    """
    by_month, rng = {}, []
    for t in config.TURBINES:
        h = load_hourly(t)
        h = h[h["n"] >= config.MIN_SAMPLES_PER_HOUR]
        month = utc_to_local(h.index).month
        by_month[t] = {int(m): float(v) for m, v in h["power"].groupby(month).mean().items()}
        rng += [h.index.min(), h.index.max()]
    by_month["farm"] = {m: float(np.mean([by_month[t][m] for t in config.TURBINES if m in by_month[t]]))
                        for m in by_month[next(iter(config.TURBINES))]}
    return {"by_month": by_month, "range": [str(min(rng))[:10], str(max(rng))[:10]]}


# ---------------------------------------------------------------- 1. погода
def _live_window(wx, turbine: str, now: pd.Timestamp) -> pd.DataFrame:
    """48 часов оперативного прогноза, начиная с ближайшего полного часа после `now` (UTC)."""
    t0 = now.floor("h") + pd.Timedelta(hours=1)
    targets = pd.date_range(t0, periods=config.HORIZON_HOURS, freq="h")
    raw = wx.live_forecast(turbine, forecast_days=3).set_index("time").sort_index()
    if raw.index.max() < targets[-1]:
        raw = wx.live_forecast(turbine, forecast_days=4).set_index("time").sort_index()
    df = raw.reindex(targets)[config.WEATHER_VARS].reset_index(names="time")
    df["target_local"] = utc_to_local(targets)
    df["lead_hours"] = np.arange(1, config.HORIZON_HOURS + 1)
    df["lead_day"] = np.where(df["lead_hours"] <= 24, 1, 2)
    df["issue_time_utc"] = t0 - pd.Timedelta(hours=1)
    return df


def fetch_weather(wx, issue_date: str | None = None, live: bool = False, now: pd.Timestamp | None = None) -> dict:
    """Получить прогноз погоды на 48 ч для обеих турбин.

    replay (live=False): архивный прогноз Open-Meteo Previous Runs, каким он был в 23:00 местного дня
    `issue_date` (сутки D+1 — запуск за 1 сутки, D+2 — за 2 суток).
    live=True: последний оперативный запуск Open-Meteo Forecast API, часы после текущего момента.

    Возвращает {'frames': {turbine: DataFrame}, 'meta': {источник, диапазон, число часов, NaN, хэш входа}}.
    """
    calls0 = wx.calls
    frames = {}
    for t in config.TURBINES:
        if live:
            now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
            frames[t] = _live_window(wx, t, now)
        else:
            frames[t] = wx.archived_forecast(t, issue_date)
    first = frames[next(iter(frames))]
    issue_utc = pd.Timestamp(first["issue_time_utc"].iloc[0])
    if live:
        issue_date = utc_to_local(pd.DatetimeIndex([issue_utc])).strftime("%Y-%m-%d")[0]
    meta = {
        "source": SOURCE_LIVE if live else SOURCE_ARCHIVE,
        "mode": "live" if live else "replay",
        "issue_date": issue_date,
        "issue_time_utc": iso_utc([issue_utc])[0],
        "issue_time_local": iso_local([issue_utc])[0],
        "target_start_local": iso_local([first["time"].min()])[0],
        "target_end_local": iso_local([first["time"].max()])[0],
        "n_hours": {t: int(len(df)) for t, df in frames.items()},
        "n_nan": {t: int(df[config.WEATHER_VARS].isna().sum().sum()) for t, df in frames.items()},
        "input_hash": input_hash(frames),
        "api_calls": int(wx.calls - calls0),
    }
    return {"frames": frames, "meta": meta}


# ---------------------------------------------------------------- 2. подготовка
def _check_range(df: pd.DataFrame, cols: list[str], lo: float, hi: float, turbine: str, name: str, notes: list) -> None:
    bad = int(((df[cols] < lo) | (df[cols] > hi)).sum().sum())
    if bad:
        notes.append(f"{turbine}: {bad} значений «{name}» вне [{lo:g}, {hi:g}] — обрезаны до границ")
        df[cols] = df[cols].clip(lo, hi)


def prepare_data(weather: dict) -> dict:
    """Проверить вход и построить признаки модели для каждой турбины.

    Проверки: нет пропусков (иначе интерполяция по времени), скорости ветра в [0, 60] м/с,
    температура в [-50, 50] °C, давление в [700, 1100] гПа, 48 часов на турбину.
    Возвращает {'features': {t: X}, 'frames': {t: очищенная погода}, 'notes': [замечания], 'meta': meta}.
    """
    features, clean, notes = {}, {}, []
    for t, df in weather["frames"].items():
        d = df.copy()
        if len(d) != config.HORIZON_HOURS:
            notes.append(f"{t}: {len(d)} часов вместо {config.HORIZON_HOURS}")
        n_nan = int(d[config.WEATHER_VARS].isna().sum().sum())
        if n_nan:
            notes.append(f"{t}: {n_nan} пропусков во входе — заполнены интерполяцией по времени")
            d[config.WEATHER_VARS] = d[config.WEATHER_VARS].interpolate(limit_direction="both")
        _check_range(d, SPEED_VARS, *SPEED_RANGE, t, "скорость ветра", notes)
        _check_range(d, ["temperature_2m"], *TEMP_RANGE, t, "температура", notes)
        _check_range(d, ["surface_pressure"], *PRESSURE_RANGE, t, "давление", notes)
        d["turbine"] = t
        X = make_features(d.set_index("time"))
        left = int(X.isna().sum().sum())
        if left:
            notes.append(f"{t}: {left} пропусков в признаках после подготовки (модель обработает их как пропуски)")
        features[t], clean[t] = X, d
    return {"features": features, "frames": clean, "notes": notes, "meta": weather["meta"]}


# ---------------------------------------------------------------- 3. модель
def run_model(prepared: dict, model: PowerModel | None = None) -> pd.DataFrame:
    """Почасовой прогноз P10/P50/P90 (доля номинала) для t1, t2 и парка `farm` в формате контракта.

    farm — среднее квантилей двух турбин (турбины коррелированы на 0.97, поэтому это близко к
    квантилям суммы); погодные колонки farm = значения t1.
    """
    model = model or load_model()
    meta, version = prepared["meta"], model_version(model)
    parts = {}
    for t, X in prepared["features"].items():
        w = prepared["frames"][t]
        p = model.predict(X).reset_index(drop=True)
        df = pd.DataFrame({
            "issue_date": meta["issue_date"],
            "issue_time_utc": iso_utc(w["issue_time_utc"]).values,
            "target_time_utc": iso_utc(w["time"]).values,
            "target_time_local": iso_local(w["time"]).values,
            "lead_hours": w["lead_hours"].astype(int).values,
            "lead_day": w["lead_day"].astype(int).values,
            "turbine": t,
            "p10": p["p10"].values, "p50": p["p50"].values, "p90": p["p90"].values,
        })
        for v in WEATHER_OUT:
            df[v] = w[v].astype(float).round(2).values
        parts[t] = df
    farm = parts["t1"].copy()
    farm["turbine"] = "farm"
    for q in ("p10", "p50", "p90"):
        farm[q] = np.mean([parts[t][q].values for t in config.TURBINES], axis=0)
    out = pd.concat([parts["t1"], parts["t2"], farm], ignore_index=True)
    for q in ("p10", "p50", "p90"):
        out[q] = out[q].astype(float).round(4)
    out["weather_source"] = meta["source"]
    out["model_version"] = version
    return out[FORECAST_COLUMNS]


# ---------------------------------------------------------------- 4. анализ
def _series(forecast: pd.DataFrame, turbine: str) -> pd.DataFrame:
    return forecast[forecast["turbine"] == turbine].sort_values("lead_hours").reset_index(drop=True)


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """Самая длинная серия True: (длина, индекс начала)."""
    best, best_start, cur, start = 0, -1, 0, 0
    for i, m in enumerate(mask):
        if m:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    return best, best_start


def _hhmm(local_iso: str) -> str:
    return local_iso[:16].replace("T", " ")


def analyze_forecast(forecast: pd.DataFrame, previous_forecast: pd.DataFrame | None = None,
                     history: dict | None = None, notes: list[str] | None = None) -> dict:
    """Проверки и показатели прогноза: диапазоны, энергия, рампы, уверенность, климатология, ревизия.

    * значения в [0, 1] и P10 ≤ P50 ≤ P90;
    * сумма P50 за 48 ч и по суткам — «часы номинала» (1 ч работы на номинале = 1.0);
    * доля часов ≥ 0.9 и ≤ 0.05, максимальный часовой скачок P50 (рампа > 0.4 — флаг);
    * средняя ширина P90−P10 (уверенность), сравнение с климатической нормой месяца по истории SCADA;
    * ревизия: сравнение с предыдущим выпуском на общих целевых часах (MAE > 0.15 — флаг).
    Возвращает dict с показателями по t1/t2/farm, флагами и итоговым статусом ok|warning.
    """
    history = history or load_history()
    flags: list[dict] = []
    for n in notes or []:
        flags.append({"code": "input_quality", "level": "warning", "message": n})

    # диапазоны и порядок квантилей
    q = forecast[["p10", "p50", "p90"]]
    out_of_range = int(((q < 0) | (q > 1)).sum().sum())
    order_viol = int(((forecast["p10"] > forecast["p50"] + 1e-9) | (forecast["p50"] > forecast["p90"] + 1e-9)).sum())
    counts = forecast.groupby("turbine").size().to_dict()
    checks = {"values_in_0_1": out_of_range == 0, "quantiles_ordered": order_viol == 0,
              "hours_per_series": {k: int(v) for k, v in counts.items()},
              "complete": all(counts.get(s, 0) == config.HORIZON_HOURS for s in SERIES)}
    if out_of_range:
        flags.append({"code": "range_violation", "level": "warning", "message": f"{out_of_range} значений вне [0, 1]"})
    if order_viol:
        flags.append({"code": "quantile_order", "level": "warning", "message": f"{order_viol} часов с нарушением P10 ≤ P50 ≤ P90"})
    if not checks["complete"]:
        flags.append({"code": "incomplete", "level": "warning", "message": f"часов по рядам: {checks['hours_per_series']}"})

    days = sorted(forecast["target_time_local"].str[:10].unique())
    metrics = {}
    for s in SERIES:
        d = _series(forecast, s)
        if d.empty:
            continue
        p50 = d["p50"].to_numpy()
        jumps = np.abs(np.diff(p50)) if len(p50) > 1 else np.array([0.0])
        i_ramp = int(np.argmax(jumps))
        months = pd.to_datetime(d["target_time_local"].str[:19]).dt.month
        clim = float(np.mean([history["by_month"][s].get(int(m), np.nan) for m in months]))
        m = {
            "energy_48h": round(float(p50.sum()), 2),
            "energy_48h_p10": round(float(d["p10"].sum()), 2),
            "energy_48h_p90": round(float(d["p90"].sum()), 2),
            "energy_by_day": {day: round(float(d.loc[d["target_time_local"].str[:10] == day, "p50"].sum()), 2) for day in days},
            "energy_by_day_p10": {day: round(float(d.loc[d["target_time_local"].str[:10] == day, "p10"].sum()), 2) for day in days},
            "energy_by_day_p90": {day: round(float(d.loc[d["target_time_local"].str[:10] == day, "p90"].sum()), 2) for day in days},
            "energy_day1": round(float(d.loc[d["lead_day"] == 1, "p50"].sum()), 2),
            "energy_day2": round(float(d.loc[d["lead_day"] == 2, "p50"].sum()), 2),
            "mean_p50": round(float(p50.mean()), 3),
            "max_p50": round(float(p50.max()), 3),
            "share_high": round(float((p50 >= HIGH_LEVEL).mean()), 3),
            "share_calm": round(float((p50 <= CALM_LEVEL).mean()), 3),
            "max_ramp": round(float(jumps[i_ramp]), 3),
            "max_ramp_time_local": _hhmm(d["target_time_local"].iloc[i_ramp + 1]) if len(p50) > 1 else None,
            "n_ramps_over_threshold": int((jumps > RAMP_THRESHOLD).sum()),
            "mean_band_p90_p10": round(float((d["p90"] - d["p10"]).mean()), 3),
            "mean_wind_100m": round(float(d["wind_speed_100m"].mean()), 2),
            "climatology_mean": round(clim, 3),
            "vs_climatology_pct": round(100 * (p50.mean() / clim - 1), 1) if clim > 0 else None,
        }
        metrics[s] = m

    farm = metrics.get("farm", {})
    if farm.get("max_ramp", 0) > RAMP_THRESHOLD:
        flags.append({"code": "ramp", "level": "warning",
                      "message": f"резкая рампа мощности парка: скачок P50 {farm['max_ramp']:.2f} номинала за час "
                                 f"к {farm['max_ramp_time_local']} (порог {RAMP_THRESHOLD}); всего скачков > порога: {farm['n_ramps_over_threshold']}"})
    if farm.get("mean_band_p90_p10", 0) > WIDE_BAND_THRESHOLD:
        flags.append({"code": "low_confidence", "level": "info",
                      "message": f"широкий интервал P10–P90 (в среднем {farm['mean_band_p90_p10']:.2f} номинала) — повышенная неопределённость"})
    fd = _series(forecast, "farm")
    if not fd.empty:
        run_len, run_start = _longest_run(fd["p50"].to_numpy() <= CALM_LEVEL)
        if run_len >= CALM_MIN_HOURS:
            flags.append({"code": "calm_window", "level": "info",
                          "message": f"штиль {run_len} ч подряд (P50 ≤ {CALM_LEVEL}) с {_hhmm(fd['target_time_local'].iloc[run_start])} — "
                                     f"кандидат на окно для ТО"})
    if farm.get("vs_climatology_pct") is not None and abs(farm["vs_climatology_pct"]) > 50:
        word = "выше" if farm["vs_climatology_pct"] > 0 else "ниже"
        flags.append({"code": "climatology", "level": "info",
                      "message": f"средняя мощность {word} климатической нормы месяца на {abs(farm['vs_climatology_pct']):.0f} % "
                                 f"({farm['mean_p50']:.2f} против {farm['climatology_mean']:.2f})"})

    # ревизия: те же целевые часы в предыдущем выпуске
    revision = None
    if previous_forecast is not None and not previous_forecast.empty:
        revision = {"previous_issue_date": str(previous_forecast["issue_date"].iloc[0])}
        for s in SERIES:
            cur = _series(forecast, s)
            prev = previous_forecast[previous_forecast["turbine"] == s]
            key_c = pd.to_datetime(cur["target_time_utc"], utc=True)
            key_p = pd.to_datetime(prev["target_time_utc"], utc=True)
            j = pd.DataFrame({"cur": cur["p50"].values}, index=key_c).join(
                pd.DataFrame({"prev": prev["p50"].values, "prev_lead": prev["lead_day"].values}, index=key_p), how="inner")
            if j.empty:
                continue
            e = j["cur"] - j["prev"]
            revision[s] = {"n_hours": int(len(j)), "mae": round(float(e.abs().mean()), 3), "bias": round(float(e.mean()), 3),
                           "max_abs": round(float(e.abs().max()), 3)}
        if "farm" in revision:
            r = revision["farm"]
            revision["significant"] = r["mae"] > REVISION_MAE_THRESHOLD
            if revision["significant"]:
                sign = "вверх" if r["bias"] > 0 else "вниз"
                flags.append({"code": "revision", "level": "warning",
                              "message": f"существенная ревизия к выпуску {revision['previous_issue_date']}: MAE {r['mae']:.2f} "
                                         f"на {r['n_hours']} общих часах, смещение {r['bias']:+.2f} ({sign}); порог {REVISION_MAE_THRESHOLD}"})
        else:
            revision = None

    status = "warning" if any(f["level"] == "warning" for f in flags) else "ok"
    return {"checks": checks, "metrics": metrics, "revision": revision, "flags": flags, "status": status,
            "days": days, "climatology_range": history.get("range")}


def analysis_brief(analysis: dict) -> dict:
    """Компактная сводка анализа для LLM и журнала (без почасовых таблиц)."""
    keep = ["energy_48h", "energy_48h_p10", "energy_48h_p90", "energy_by_day", "energy_day1", "energy_day2", "mean_p50",
            "share_high", "share_calm", "max_ramp", "max_ramp_time_local", "mean_band_p90_p10", "mean_wind_100m",
            "climatology_mean", "vs_climatology_pct"]
    return {"status": analysis["status"], "checks": analysis["checks"],
            "metrics": {s: {k: m.get(k) for k in keep} for s, m in analysis["metrics"].items()},
            "revision": analysis["revision"], "flags": analysis["flags"]}


# ---------------------------------------------------------------- 5. нарратив по шаблону
def _pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def template_narrative(analysis: dict, meta: dict | None = None) -> str:
    """Детерминированный текст отчёта (3–6 предложений с числами) — когда LLM недоступна."""
    f = analysis["metrics"]["farm"]
    days = analysis["days"]
    d1 = f["energy_by_day"].get(days[0], f["energy_day1"]) if days else f["energy_day1"]
    s = [f"Ожидаемая выработка парка за 48 ч — {f['energy_48h']:.1f} часа номинала "
         f"(P10–P90: {f['energy_48h_p10']:.1f}–{f['energy_48h_p90']:.1f}), средняя мощность {_pct(f['mean_p50'])} номинала "
         f"при среднем прогнозном ветре {f['mean_wind_100m']:.1f} м/с на 100 м."]
    if len(days) >= 2:
        d2 = f["energy_by_day"].get(days[1], f["energy_day2"])
        trend = "рост" if d2 > d1 * 1.15 else ("снижение" if d2 < d1 * 0.85 else "примерно тот же уровень")
        s.append(f"По суткам: {days[0]} — {d1:.1f} ч.н., {days[1]} — {d2:.1f} ч.н. ({trend} ко вторым суткам).")
    if f.get("vs_climatology_pct") is not None:
        word = "выше" if f["vs_climatology_pct"] >= 0 else "ниже"
        s.append(f"Это {word} климатической нормы месяца ({_pct(f['climatology_mean'])}) на {abs(f['vs_climatology_pct']):.0f} %; "
                 f"часов с мощностью ≥ 90 % — {_pct(f['share_high'])}, почти без генерации (≤ 5 %) — {_pct(f['share_calm'])}.")
    s.append(f"Максимальный часовой скачок P50 — {f['max_ramp']:.2f} номинала ({f['max_ramp_time_local']}), "
             f"средняя ширина интервала P10–P90 — {f['mean_band_p90_p10']:.2f}.")
    rev = analysis.get("revision")
    if rev and "farm" in rev:
        r = rev["farm"]
        verdict = "существенная ревизия, прогноз погоды заметно изменился" if rev.get("significant") else "прогноз устойчив"
        s.append(f"К выпуску {rev['previous_issue_date']} на {r['n_hours']} общих часах MAE {r['mae']:.2f}, "
                 f"смещение {r['bias']:+.2f} — {verdict}.")
    warn = [x for x in analysis["flags"] if x["level"] == "warning"]
    s.append(f"Флагов, требующих внимания диспетчера: {len(warn)}." if warn else "Проверки пройдены, флагов, требующих внимания, нет.")
    return " ".join(s[:6])


# ---------------------------------------------------------------- 6. запись результатов
def save_forecast(issue_date: str, forecast: pd.DataFrame, out_dir: Path | None = None, name: str | None = None) -> Path:
    """Сохранить прогноз в формате контракта: outputs/forecasts/<issue_date>.csv (или out_dir/<name>.csv)."""
    out_dir = Path(out_dir or config.OUTPUTS_DIR / "forecasts")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name or issue_date}.csv"
    forecast[FORECAST_COLUMNS].to_csv(path, index=False)
    return path


def _md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _energy_cell(m: dict, day: str | None = None) -> str:
    if day is None:
        return f"**{m['energy_48h']:.1f}** [{m['energy_48h_p10']:.1f}–{m['energy_48h_p90']:.1f}]"
    return f"{m['energy_by_day'][day]:.1f} [{m['energy_by_day_p10'][day]:.1f}–{m['energy_by_day_p90'][day]:.1f}]"


DECISION_RU = {"accept": "принять прогноз", "recalculate": "пересчитать (повторный запрос входа)", "flag": "принять с флагом для диспетчера"}


def write_report(issue_date: str, forecast: pd.DataFrame, analysis: dict, narrative: str, *, meta: dict | None = None,
                 decision: dict | None = None, trace: list[dict] | None = None, out_dir: Path | None = None,
                 name: str | None = None) -> Path:
    """Markdown-отчёт агента: outputs/reports/<issue_date>.md (или out_dir/<name>.md).

    Разделы: источник и время выпуска, суточные суммы по турбинам и парку, ключевые показатели, флаги,
    комментарий (LLM или шаблон), ход работы агента, почасовая таблица парка на 48 ч.
    """
    meta = meta or {}
    decision = decision or {}
    out_dir = Path(out_dir or config.OUTPUTS_DIR / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name or issue_date}.md"
    m, days = analysis["metrics"], analysis["days"]
    f = m["farm"]
    live = meta.get("mode") == "live"
    first = forecast.iloc[0]

    L = [f"# {'Оперативный прогноз' if live else 'Прогноз'} выработки ВЭС — выпуск {issue_date}", ""]
    if live:
        L += ["> **Оперативный прогноз** по последнему запуску Open-Meteo Forecast API (без архивного лага), "
              "сформирован в момент запуска агента. Для ретроспективной проверки используйте режим `replay`.", ""]
    else:
        L += ["> Ретроспективный прогон: прогноз сформирован так, как если бы он выпускался в 23:00 местного времени "
              f"{issue_date}, только по прогнозам погоды, доступным на тот момент (Open-Meteo Previous Runs: "
              "сутки D+1 — запуск за 1 сутки, D+2 — запуск за 2 суток).", ""]
    L += [f"- Время выпуска: {first['issue_time_utc']} (местное {meta.get('issue_time_local', '—')})",
          f"- Горизонт: {_hhmm(forecast['target_time_local'].min())} — {_hhmm(forecast['target_time_local'].max())} "
          f"(местное, {config.LOCAL_TZ_NAME}), {config.HORIZON_HOURS} ч",
          f"- Источник погоды: `{first['weather_source']}`, хэш входа `{meta.get('input_hash', '—')}`",
          f"- Модель: `{first['model_version']}` (HistGradientBoosting, квантили P10/P50/P90, доля номинала)",
          f"- Решение агента: **{decision.get('decision', '—')}** — {DECISION_RU.get(decision.get('decision'), '')}; "
          f"статус проверок: **{analysis['status']}**; LLM: {'да, ' + decision.get('model', '') if decision.get('llm_used') else 'нет (детерминированный режим)'}",
          ""]

    L += ["## Выработка по суткам", "",
          "Часы номинала: сумма почасовой нормированной мощности (1.0 = 1 ч на номинале). "
          "P50 и в скобках [сумма P10 – сумма P90].", ""]
    rows = [[f"{day} (D+{i + 1})"] + [_energy_cell(m[s], day) for s in SERIES] for i, day in enumerate(days)]
    rows.append(["**Итого 48 ч**"] + [_energy_cell(m[s]) for s in SERIES])
    L += [_md_table(["Сутки (местные)", "Турбина 1", "Турбина 2", "Парк (среднее)"], rows), ""]

    L += ["## Ключевые показатели (парк)", "",
          f"- Средняя мощность P50: {f['mean_p50']:.2f} номинала; максимум {f['max_p50']:.2f}; средний ветер 100 м {f['mean_wind_100m']:.1f} м/с",
          f"- Климатическая норма месяца (SCADA {analysis.get('climatology_range', ['', ''])[0]}…{analysis.get('climatology_range', ['', ''])[1]}): "
          f"{f['climatology_mean']:.2f} → отклонение {f['vs_climatology_pct']:+.0f} %",
          f"- Часы ≥ 0.9 номинала: {_pct(f['share_high'])}; часы ≤ 0.05: {_pct(f['share_calm'])}",
          f"- Макс. часовой скачок P50: {f['max_ramp']:.2f} (к {f['max_ramp_time_local']}), скачков > {RAMP_THRESHOLD}: {f['n_ramps_over_threshold']}",
          f"- Средняя ширина P90−P10: {f['mean_band_p90_p10']:.2f} (чем уже, тем увереннее прогноз)"]
    rev = analysis.get("revision")
    if rev and "farm" in rev:
        r = rev["farm"]
        L.append(f"- Ревизия к выпуску {rev['previous_issue_date']} ({r['n_hours']} общих часов): MAE {r['mae']:.3f}, "
                 f"bias {r['bias']:+.3f}, max |Δ| {r['max_abs']:.2f} → {'**существенная**' if rev.get('significant') else 'в норме'} "
                 f"(порог {REVISION_MAE_THRESHOLD})")
    else:
        L.append("- Ревизия: предыдущего выпуска на те же часы нет")
    c = analysis["checks"]
    L += [f"- Проверки: значения в [0, 1] — {'да' if c['values_in_0_1'] else 'НЕТ'}; P10 ≤ P50 ≤ P90 — "
          f"{'да' if c['quantiles_ordered'] else 'НЕТ'}; 48 ч по каждому ряду — {'да' if c['complete'] else 'НЕТ'}", ""]

    L += ["## Флаги", ""]
    L += [f"- [{x['level']}] `{x['code']}`: {x['message']}" for x in analysis["flags"]] or ["- нет"]
    L.append("")

    L += ["## Комментарий агента", "", narrative.strip(), ""]
    if decision.get("reasoning"):
        L += [f"*Обоснование решения:* {decision['reasoning'].strip()}", ""]
    if decision.get("recalc"):
        rc = decision["recalc"]
        L += [f"*Пересчёт:* вход {'обновился' if rc.get('input_changed') else 'не изменился'} "
              f"(хэш {rc.get('old_hash')} → {rc.get('new_hash')}), "
              f"{'прогноз пересчитан' if rc.get('input_changed') else 'прогноз подтверждён'}, max |ΔP50| = {rc.get('max_abs_diff', 0):.4f}", ""]

    if trace:
        L += ["## Ход работы агента", ""]
        L += [_md_table(["#", "Инструмент", "Кто вызвал", "Статус", "Время, с", "Итог"],
                        [[i + 1, f"`{t['tool']}`", "LLM" if t.get("llm_used") else "правила", t["status"], f"{t['duration_s']:.2f}",
                          str(t.get("summary", "")).replace("|", "/")[:160]] for i, t in enumerate(trace)]), ""]

    fd = _series(forecast, "farm")
    L += ["## Почасовой прогноз парка (48 ч)", ""]
    L += [_md_table(["Время (местное)", "Упреждение, ч", "Ветер 100 м, м/с", "P10", "P50", "P90"],
                    [[_hhmm(r.target_time_local), r.lead_hours, f"{r.wind_speed_100m:.1f}", f"{r.p10:.2f}", f"{r.p50:.2f}", f"{r.p90:.2f}"]
                     for r in fd.itertuples()]), ""]
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 7. сводки для журнала
def weather_summary(meta: dict) -> str:
    return (f"{meta['source']}: {meta['target_start_local'][:16]}…{meta['target_end_local'][:16]}, часов "
            f"{'/'.join(str(v) for v in meta['n_hours'].values())}, NaN {sum(meta['n_nan'].values())}, хэш {meta['input_hash']}")


def forecast_summary(forecast: pd.DataFrame) -> str:
    fd = forecast[forecast["turbine"] == "farm"]
    return f"{len(forecast)} строк; парк: P50 ср. {fd['p50'].mean():.3f}, сумма {fd['p50'].sum():.1f} ч.н."


def clean_text(s: str, limit: int = 2000) -> str:
    """Нормализовать текст от LLM: убрать управляющие символы и лишние пробелы, ограничить длину."""
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", str(s))
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s[:limit]
