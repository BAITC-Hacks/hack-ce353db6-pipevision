"""Reproduce R3 diagnostics from the saved backtest, without fitting a model.

Run: PYTHONPATH=src python scripts/make_error_analysis.py
Writes five figures, machine-readable tables and docs/research/R3_error_analysis.md.
Weather direction comes exclusively from the checked-in Previous Runs cache.
"""
from __future__ import annotations

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_agent import config
from wind_agent.data import utc_to_local
from wind_agent.weather import WeatherClient

FIG_DIR = config.OUTPUTS_DIR / "figures"
DATA_DIR = config.ROOT / "docs/research/data"
REPORT = config.ROOT / "docs/research/R3_error_analysis.md"
KEYS = ["time", "turbine", "lead_day"]
WIND_LABELS = ["<3", "3–6", "6–9", "9–12", "≥12"]
POWER_LABELS = ["≤0,05", "0,05–0,3", "0,3–0,7", "0,7–0,95", ">0,95"]
RAMP_LABELS = {"steady": "Без рампы", "up": "Рост ≥0,3/ч", "down": "Спад ≤−0,3/ч", "unknown": "Нет предыдущего часа"}
SECTORS = ["С", "ССВ", "СВ", "ВСВ", "В", "ВЮВ", "ЮВ", "ЮЮВ", "Ю", "ЮЮЗ", "ЮЗ", "ЗЮЗ", "З", "ЗСЗ", "СЗ", "ССЗ"]
COLORS = ["#2563eb", "#f97316"]
RAMP_THRESHOLD = .3 - 1e-12  # Include mathematical 0.3 despite subtraction roundoff.


def add_ramps(frame: pd.DataFrame) -> pd.DataFrame:
    """A ramp requires an exactly adjacent hour within the same turbine and lead.

    Actual ramps remain valid across midnight. Predicted changes are compared only
    within a single nominal issue date, never across two different daily releases.
    Missing predecessors are unknown, not steady hours.
    """
    b = frame.sort_values(["turbine", "lead_day", "time"]).copy()
    group = b.groupby(["turbine", "lead_day"], sort=False)
    adjacent = group.time.diff().eq(pd.Timedelta(hours=1))
    b["actual_delta"] = group.power.diff().where(adjacent)
    same_issue = b.issue_date.eq(group.issue_date.shift())
    b["predicted_delta"] = group.p50.diff().where(adjacent & same_issue)
    delta = b.actual_delta
    b["ramp"] = np.select([delta.isna(), delta >= RAMP_THRESHOLD, delta <= -RAMP_THRESHOLD],
                          ["unknown", "up", "down"], default="steady")
    return b


def prepare_predictions(frame: pd.DataFrame, directions: pd.DataFrame) -> pd.DataFrame:
    b = frame.copy()
    b["time"] = pd.to_datetime(b.time, utc=True)
    if b.duplicated(KEYS).any():
        raise ValueError("Duplicate forecast key: time/turbine/lead_day")
    values = b[["power", "p10", "p50", "p90", "wind_speed_100m", "wind_meas"]]
    if not np.isfinite(values).all().all():
        raise ValueError("Non-finite required backtest values")
    if not b.lead_day.isin([1, 2]).all():
        raise ValueError("R3 expects the original 23:00 issue, previous_day1/2 protocol")
    if not ((b.p10 <= b.p50) & (b.p50 <= b.p90) & (b.p10 >= 0) & (b.p90 <= 1)).all():
        raise ValueError("Invalid predictive intervals")
    if not (b.power.between(0, 1) & (b.wind_speed_100m >= 0)).all():
        raise ValueError("Invalid power or wind range")
    d = directions.copy()
    d["time"] = pd.to_datetime(d.time, utc=True)
    b = b.merge(d, on=KEYS, how="left", validate="one_to_one")
    if not np.isfinite(b.direction).all():
        raise ValueError("Missing direction in offline weather join")
    # Same physical measurement must not differ between the two forecast leads.
    if (b.groupby(["time", "turbine"]).power.nunique() > 1).any():
        raise ValueError("Inconsistent actual power between lead days")
    b["local_time"] = utc_to_local(pd.DatetimeIndex(b.time)).to_numpy()
    b["hour"] = b.local_time.dt.hour
    b["date"] = b.local_time.dt.strftime("%Y-%m-%d")
    b["month"] = b.local_time.dt.strftime("%Y-%m")
    b["issue_date"] = b.local_time.dt.normalize() - pd.to_timedelta(b.lead_day, unit="D")
    # Derived contractual horizon, not an independently recorded NWP model run age.
    b["horizon_hour"] = (b.lead_day - 1) * 24 + b.hour + 1
    b["error"] = b.p50 - b.power
    b["abs_error"] = b.error.abs()
    b["squared_error"] = b.error**2
    b["covered"] = b.power.between(b.p10, b.p90)
    b["below_p10"] = b.power < b.p10
    b["above_p90"] = b.power > b.p90
    b["width"] = b.p90 - b.p10
    b["wind_regime"] = pd.cut(b.wind_speed_100m, [0, 3, 6, 9, 12, np.inf], labels=WIND_LABELS, right=False)
    b["power_regime"] = pd.cut(b.power, [-np.inf, .05, .3, .7, .95, np.inf], labels=POWER_LABELS)
    b["sector_i"] = np.floor(((b.direction % 360) + 11.25) / 22.5).astype(int) % 16
    b["sector"] = pd.Categorical([SECTORS[i] for i in b.sector_i], categories=SECTORS, ordered=True)
    b["wind_difference"] = b.wind_speed_100m - b.wind_meas
    b["wind_difference_abs"] = pd.cut(b.wind_difference.abs(), [0, 1, 2, 4, np.inf],
                                           labels=["<1", "1–2", "2–4", "≥4"], right=False)
    return add_ramps(b)


def load_predictions():
    frame = pd.read_csv(config.OUTPUTS_DIR / "backtest_predictions.csv")
    times = pd.to_datetime(frame.time, utc=True)
    start, end = times.min().strftime("%Y-%m-%d"), times.max().strftime("%Y-%m-%d")
    weather, parts = WeatherClient(offline=True), []
    for turbine in sorted(frame.turbine.unique()):
        raw = weather.previous_runs(turbine, start, end)
        for lead in sorted(frame.lead_day.unique()):
            d = raw[["time", f"wind_direction_100m_previous_day{lead}"]].rename(
                columns={f"wind_direction_100m_previous_day{lead}": "direction"})
            parts.append(d.assign(turbine=turbine, lead_day=lead))
    return prepare_predictions(frame, pd.concat(parts, ignore_index=True))


def summarize(b, by):
    table = b.groupby(by, observed=True, sort=True).agg(
        n=("error", "size"), mae=("abs_error", "mean"), mse=("squared_error", "mean"),
        bias=("error", "mean"), coverage_pct=("covered", "mean"), below_p10_pct=("below_p10", "mean"),
        above_p90_pct=("above_p90", "mean"), mean_width=("width", "mean"), mean_actual=("power", "mean"),
        absolute_error_sum=("abs_error", "sum"), p90_absolute_error=("abs_error", lambda s: s.quantile(.9)))
    table["rmse"] = np.sqrt(table.pop("mse"))
    table["row_share_pct"] = 100 * table.n / len(b)
    table["error_share_pct"] = 100 * table.absolute_error_sum / b.abs_error.sum()
    for col in ["coverage_pct", "below_p10_pct", "above_p90_pct"]:
        table[col] *= 100
    return table


def ramp_detection(b):
    rows = []
    for lead, g in b.groupby("lead_day"):
        g = g[g.predicted_delta.notna()]
        truth, pred = g.actual_delta, g.predicted_delta
        actual_ramp, predicted_ramp = truth.abs() >= RAMP_THRESHOLD, pred.abs() >= RAMP_THRESHOLD
        correct = actual_ramp & predicted_ramp & (np.sign(truth) == np.sign(pred))
        rows.append({"lead_day": int(lead), "eligible_transitions": len(g),
                     "actual_ramps": int(actual_ramp.sum()), "predicted_ramps": int(predicted_ramp.sum()),
                     "correct_direction_ramps": int(correct.sum()),
                     "directional_recall_pct": 100 * correct.sum() / actual_ramp.sum() if actual_ramp.any() else np.nan,
                     "directional_precision_pct": 100 * correct.sum() / predicted_ramp.sum() if predicted_ramp.any() else np.nan,
                     "delta_mae": float((pred - truth).abs().mean())})
    return pd.DataFrame(rows).set_index("lead_day")


def make_tables(b):
    dimensions = {"turbine_lead": ["turbine", "lead_day"], "wind": "wind_regime", "power": "power_regime",
                  "hour": "hour", "hour_lead": ["hour", "lead_day"], "horizon": "horizon_hour",
                  "month": "month", "month_lead": ["month", "lead_day"], "sector": "sector",
                  "ramps": "ramp", "ramps_lead": ["ramp", "lead_day"], "daily": "date",
                  "month_wind": ["month", "wind_regime"], "wind_difference": "wind_difference_abs"}
    tables = {name: summarize(b, by) for name, by in dimensions.items()}
    tables["overall"] = summarize(b.assign(group="overall"), "group")
    tables["lead"] = summarize(b, "lead_day")
    tables["ramp_detection"] = ramp_detection(b)
    for name, table in tables.items():
        table.to_csv(DATA_DIR / f"R3_{name}.csv")
    cols = KEYS + ["local_time", "horizon_hour", "power", "p10", "p50", "p90", "error",
                   "wind_speed_100m", "wind_meas", "direction", "ramp"]
    b.nlargest(20, "abs_error")[cols].to_csv(DATA_DIR / "R3_largest_errors.csv", index=False)
    return tables


def save_figure(fig, filename):
    fig.savefig(FIG_DIR / filename, dpi=170, facecolor="white")
    plt.close(fig)


def bar_labels(ax, values, fmt=".3f"):
    for i, value in enumerate(values):
        ax.annotate(format(value, fmt), (i, value), ha="center", va="bottom" if value >= 0 else "top",
                    xytext=(0, 4 if value >= 0 else -4), textcoords="offset points", fontsize=9)
    ax.margins(y=.2)


def make_figures(b, t):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.alpha": .18})
    wind = t["wind"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    fig.suptitle("R3 · Основной вклад в ошибку — прогнозный ветер 3–9 м/с\nНурлы · декабрь 2025 — январь 2026 · модель после R1", fontsize=15)
    ax = axs[0, 0]; ax.bar(wind.index.astype(str), wind.mae, color="#2563eb"); bar_labels(ax, wind.mae)
    ax.set(title="MAE по прогнозной скорости", ylabel="Доля номинальной мощности", xlabel="Прогноз на 100 м, м/с")
    ax = axs[0, 1]; x = np.arange(len(wind))
    ax.bar(x - .18, wind.row_share_pct, .36, label="Доля строк", color="#94a3b8")
    ax.bar(x + .18, wind.error_share_pct, .36, label="Доля суммы |ошибок|", color="#f97316")
    ax.set_xticks(x, wind.index); ax.set(title="Частота режима и вклад в ошибку", ylabel="%", xlabel="Прогноз на 100 м, м/с"); ax.legend(fontsize=9)
    ax = axs[1, 0]; ax.bar(wind.index.astype(str), wind.bias, color="#14b8a6"); bar_labels(ax, wind.bias, "+.3f")
    ax.axhline(0, color="#334155", lw=.7); ax.set(title="Bias = P50 − факт", ylabel="Знак + означает завышение", xlabel="Прогноз на 100 м, м/с")
    ax = axs[1, 1]; ax.plot(wind.index.astype(str), wind.coverage_pct, "o-", color="#7c3aed", lw=2)
    ax.axhline(80, ls="--", color="#64748b", label="Номинальные 80%")
    for i, (_, row) in enumerate(wind.iterrows()):
        ax.annotate(f"{row.coverage_pct:.1f}%\nn={int(row.n)}", (i, row.coverage_pct), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))
    ax.set(title="Покрытие P10–P90", ylabel="% строк внутри интервала", ylim=(35, 100), xlabel="Прогноз на 100 м, м/с"); ax.legend(loc="lower left")
    save_figure(fig, "error_analysis.png")

    power = t["power"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    fig.suptitle("R3 · При малом факте прогноз завышен, у номинала — занижен\nГруппировка по факту: ретроспективная диагностика, не правило онлайн-коррекции", fontsize=14)
    ax = axs[0]; ax.bar(power.index.astype(str), power.bias, color=np.where(power.bias >= 0, "#f97316", "#2563eb")); bar_labels(ax, power.bias, "+.3f")
    ax.axhline(0, color="#334155", lw=.8); ax.set(title="Средняя ошибка P50 − факт", ylabel="Доля номинала", xlabel="Фактическая мощность / номинал")
    ax = axs[1]; ax.plot(power.index.astype(str), power.coverage_pct, "o-", color="#7c3aed", lw=2)
    ax.axhline(80, ls="--", color="#64748b", label="Номинальные 80%")
    for i, (_, row) in enumerate(power.iterrows()):
        ax.annotate(f"{row.coverage_pct:.1f}%\nn={int(row.n)}", (i, row.coverage_pct), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))
    ax.set(title="Покрытие по уровню факта", ylabel="%", ylim=(35, 100), xlabel="Фактическая мощность / номинал"); ax.legend(loc="lower left")
    save_figure(fig, "R3_power_bias.png")

    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    fig.suptitle("R3 · Время суток, горизонт и месяц\nМестное время UTC+5 · горизонт восстановлен для выпуска в 23:00", fontsize=15)
    for lead, color in zip([1, 2], COLORS):
        h = t["hour_lead"].xs(lead, level="lead_day")
        for ax, metric in [(axs[0, 0], "mae"), (axs[0, 1], "bias")]:
            ax.plot(h.index, h[metric], "o-", ms=3, color=color, label=f"Лаг {lead}")
            ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("Местный час"); ax.legend()
    axs[0, 0].set(title="MAE по часу суток", ylabel="Доля номинала", ylim=(0, .3))
    axs[0, 1].set(title="Bias по часу суток", ylabel="P50 − факт"); axs[0, 1].axhline(0, color="#64748b", lw=.7)
    h = t["horizon"]; axs[1, 0].plot(h.index, h.mae, ".-", color="#2563eb")
    axs[1, 0].axvline(24.5, ls="--", color="#64748b"); axs[1, 0].set(title="MAE по договорному горизонту 1–48 ч", xlabel="Час после выпуска", ylabel="Доля номинала", xticks=[1, 6, 12, 18, 24, 30, 36, 42, 48], ylim=(0, .3))
    ax = axs[1, 1]; month = t["month_lead"].mae.unstack("lead_day"); x = np.arange(len(month))
    for lead, offset, color in [(1, -.18, COLORS[0]), (2, .18, COLORS[1])]:
        bars = ax.bar(x + offset, month[lead], .36, label=f"Лаг {lead}", color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xticks(x, month.index); ax.set(title="MAE по месяцу и лагу", ylabel="Доля номинала", ylim=(0, .3)); ax.legend()
    save_figure(fig, "R3_time_horizon.png")

    ramps = t["ramps"].reindex(["steady", "up", "down"])
    fig, axs = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
    fig.suptitle("R3 · Часовые рампы сглаживаются прогнозом\n|Δ факта| ≥0,3 номинала за 1 ч · переходы через пропуски исключены", fontsize=15)
    names = [RAMP_LABELS[k].replace(" ", "\n", 1) + f"\nn={int(ramps.loc[k, 'n'])}" for k in ramps.index]
    for ax, metric, title in zip(axs, ["mae", "bias", "coverage_pct"], ["MAE", "Bias = P50 − факт", "Покрытие P10–P90, %"]):
        values = ramps[metric]; ax.bar(names, values, color=["#94a3b8", "#2563eb", "#f97316"])
        bar_labels(ax, values, ".1f" if metric == "coverage_pct" else "+.3f" if metric == "bias" else ".3f")
        ax.set_title(title); ax.axhline(0, color="#64748b", lw=.7)
        if metric == "coverage_pct":
            ax.axhline(80, color="#64748b", ls="--"); ax.set_ylim(0, 105)
    save_figure(fig, "R3_ramps.png")

    # Worst full local day by pooled MAE. Plot one turbine and both leads; never
    # pick a good example by hand or join lines across excluded observations.
    day = t["daily"].query("n >= 80").mae.idxmax()
    center = pd.Timestamp(day)
    part = b[(b.turbine == "t1") & b.local_time.between(center - pd.Timedelta(days=1), center + pd.Timedelta(days=2), inclusive="left")]
    fig, axs = plt.subplots(3, 1, figsize=(13, 8), sharex=True, layout="constrained")
    fig.suptitle(f"R3 · Трудный эпизод вокруг {day} · T1\nДень выбран по максимальному среднему MAE обеих турбин и лагов", fontsize=14)
    index = pd.date_range(center - pd.Timedelta(days=1), center + pd.Timedelta(days=2) - pd.Timedelta(hours=1), freq="h")
    for lead, color, ax in zip([1, 2], COLORS, axs[:2]):
        g = part[part.lead_day == lead].set_index("local_time").reindex(index)
        ax.fill_between(g.index, g.p10, g.p90, alpha=.17, color=color, label="P10–P90")
        ax.plot(g.index, g.power, color="#0f172a", lw=1.7, label="Факт")
        ax.plot(g.index, g.p50, color=color, lw=1.5, label=f"P50 · лаг {lead}")
        ax.set(ylabel="Доля номинала", ylim=(-.04, 1.04)); ax.legend(ncol=3, loc="upper right", fontsize=9)
    for lead, color in zip([1, 2], COLORS):
        g = part[part.lead_day == lead].set_index("local_time").reindex(index)
        axs[2].plot(g.index, g.wind_speed_100m, color=color, label=f"Прогноз 100 м · лаг {lead}")
        if lead == 1:
            axs[2].plot(g.index, g.wind_meas, color="#0f172a", label="Замер на турбине")
    axs[2].set(ylabel="Ветер, м/с", xlabel="Местное время UTC+5"); axs[2].legend(ncol=3, fontsize=9)
    for ax in axs:
        ax.axvspan(center, center + pd.Timedelta(days=1), color="#64748b", alpha=.07)
    axs[2].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12])); axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    save_figure(fig, "R3_worst_episode.png")
    return day


def markdown(table, index_label):
    columns = {"n": "Строк", "row_share_pct": "Доля, %", "mae": "MAE", "rmse": "RMSE", "bias": "Bias",
               "coverage_pct": "Покрытие, %", "error_share_pct": "Вклад в абс. ошибку, %"}
    lines = ["| " + index_label + " | " + " | ".join(columns.values()) + " |", "|" + "---|" * (len(columns) + 1)]
    for key, row in table.iterrows():
        label = " / ".join(map(str, key)) if isinstance(key, tuple) else str(key)
        values = [str(int(row[c])) if c == "n" else f"{row[c]:+.4f}" if c == "bias"
                  else f"{row[c]:.1f}" if c.endswith("pct") else f"{row[c]:.4f}" for c in columns]
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(b, t, day):
    # Narrative is generated alongside the numeric evidence; no stale constants
    # copied from the pre-R1 model. Keep causal explanations explicitly tentative.
    overall = t["overall"].iloc[0]
    wind, power, lead = t["wind"], t["power"], t["lead"]
    core = wind.loc[["3–6", "6–9"]]
    corr = b.error.corr(b.wind_difference)
    known = b[b.ramp != "unknown"]
    ramps = known[known.ramp != "steady"]
    unknown = int((b.ramp == "unknown").sum())
    top_hour, bottom_hour = int(t["hour"].mae.idxmax()), int(t["hour"].mae.idxmin())
    det = t["ramp_detection"]
    episode = b[(b.date == day) & (b.turbine == "t1") & (b.lead_day == 1)]
    worst_row = episode.loc[episode.abs_error.idxmax()]
    meta = {
        "prediction_sha256": hashlib.sha256((config.OUTPUTS_DIR / "backtest_predictions.csv").read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256((config.OUTPUTS_DIR / "backtest_metrics.json").read_bytes()).hexdigest(),
        "n_rows": len(b), "n_turbine_hours": len(b[["time", "turbine"]].drop_duplicates()),
        "n_unique_timestamps": b.time.nunique(), "local_start": str(b.local_time.min()), "local_end": str(b.local_time.max()),
        "unknown_ramp_rows": unknown, "valid_ramp_transitions": len(known), "actual_ramp_rows": len(ramps),
        "power_error_wind_difference_correlation": corr, "worst_day": day,
    }
    (DATA_DIR / "R3_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    ramp_table = t["ramps"].reindex(["steady", "up", "down", "unknown"]).rename(index=RAMP_LABELS)
    worst = t["daily"].nlargest(5, "mae")
    text = f'''# R3 — где ошибается модель после R1

**Результат:** MAE **{overall.mae:.5f}**, RMSE **{overall.rmse:.5f}**, bias **{overall.bias:+.5f}**, покрытие P10–P90 **{overall.coverage_pct:.2f}%**. Анализирует сохранённый бэктест после R1; не переобучает модель и не меняет калибровку R2.

## 1. Данные и границы выводов

Источник: `outputs/backtest_predictions.csv`, проверен на совпадение агрегатов с `outputs/backtest_metrics.json`. Тест по прежнему протоколу 01.12.2025–31.01.2026, обучение до 30.11.2025, калибровка октябрь–ноябрь. Реальные крайние отметки теста: **{b.local_time.min():%d.%m.%Y %H:%M} — {b.local_time.max():%d.%m.%Y %H:%M} UTC+5**. Погода для направления берётся офлайн из того же Previous Runs cache, по ключу время × турбина × лаг; соединение строго 1:1.

**{len(b)} строк** — это {meta['n_turbine_hours']} различных «турбина × час» и {meta['n_unique_timestamps']} различных временных отметок. Один и тот же факт повторяется для двух лагов. Это не {len(b)} независимых часов; соседние часы и турбины также связаны. Таблицы описательные, доверительные интервалы и статистическая значимость не заявляются. Уже исключены часы с недостатком SCADA и признаками простоя/ограничения по фильтру основного бэктеста: выводы не покрывают все эксплуатационные отказы и не распространяются на лето или другие ВЭС.

Ошибка — `P50 − факт`, единица — доля номинальной мощности. Положительный bias означает завышение. «Вклад» — доля суммы абсолютных ошибок, а не финансового ущерба. Прогнозный режим доступен до наступления часа; группы по факту, рампам и расхождению с замером — только ретроспективная диагностика.

{markdown(t['turbine_lead'], 'Турбина / лаг')}

## 2. Ветер: где сосредоточена ошибка

![Режимы ветра и вклад в ошибку](../../outputs/figures/error_analysis.png)

{markdown(wind, 'Прогноз 100 м, м/с')}

Интервалы скорости **[0,3), [3,6), [6,9), [9,12), [12,+∞)**. Название «≥12» означает сильный прогнозный ветер, а не гарантированный номинал турбины. Диапазон **3–9 м/с** занимает **{core.row_share_pct.sum():.1f}%** строк и даёт **{core.error_share_pct.sum():.1f}%** суммы абсолютных ошибок. Максимальный MAE — в **{wind.mae.idxmax()} м/с** ({wind.mae.max():.4f}); это первый кандидат для дальнейших проверок качества прогноза.

При прогнозе <3 м/с bias **{wind.iloc[0].bias:+.4f}**, MAE **{wind.iloc[0].mae:.4f}**: утверждение «модель систематически занижает штиль» этими данными не подтверждается. Но слабый прогнозный ветер и низкая фактическая мощность — разные группы.

## 3. Смещение по факту и интервалы

![Смещение и покрытие по факту](../../outputs/figures/R3_power_bias.png)

{markdown(power, 'Фактическая мощность / номинал')}

При факте ≤0,05 среднее завышение **{power.iloc[0].bias:+.4f}**, при факте >0,95 — занижение **{power.iloc[-1].bias:+.4f}**. Такой рисунок совместим со сглаживанием экстремумов при неопределённом ветре и с условным отбором по самой целевой переменной. Он **не доказывает** ни неисправность, ни безошибочность модели. Нельзя автоматически вычитать эти bias в рабочем прогнозе: будущий факт неизвестен.

Покрытие в целом **{overall.coverage_pct:.2f}%** вместо номинальных 80%; факт ниже P10 в **{overall.below_p10_pct:.2f}%**, выше P90 в **{overall.above_p90_pct:.2f}%** строк. Средняя ширина **{overall.mean_width:.5f}**. В группе факта >0,95 покрытие **{power.iloc[-1].coverage_pct:.1f}%**, поэтому прежнее утверждение о максимальном покрытии у потолка неверно. Покрытие для сильного прогнозного ветра ≥12 м/с — **{wind.iloc[-1].coverage_pct:.1f}%**, но число строк здесь всего **{int(wind.iloc[-1].n)}**. Калибровка не изменяется в R3; разрезы служат диагностикой.

## 4. Местный час, лаг и месяц

![Время, горизонт и месяц](../../outputs/figures/R3_time_horizon.png)

{markdown(lead, 'Лаг, сутки')}

{markdown(t['month_lead'], 'Месяц / лаг')}

MAE второго лага выше первого на **{100 * (lead.loc[2, 'mae'] / lead.loc[1, 'mae'] - 1):.1f}%**. В январе общий MAE ниже, чем в декабре (**{t['month'].iloc[1].mae:.4f} против {t['month'].iloc[0].mae:.4f}**), но положительный bias выше (**{t['month'].iloc[1].bias:+.4f} против {t['month'].iloc[0].bias:+.4f}**). На объединённой выборке максимум по местному часу — **{top_hour:02d}:00** ({t['hour'].loc[top_hour, 'mae']:.4f}), минимум — **{bottom_hour:02d}:00** ({t['hour'].loc[bottom_hour, 'mae']:.4f}). Каждый час представлен примерно двумя месяцами, поэтому это не установленный суточный закон.

Горизонт восстановлен по договорному выпуску в 23:00: `h = 24*(lead_day−1) + local_hour + 1`. В пределах одного лага горизонт и местный час связаны однозначно: нельзя отделить влияние возраста прогноза от времени суток только этими графиками. Это не утверждение о точном возрасте исходного запуска NWP. Месячный разрез — по местной дате, границы основного теста сохранены. [Все 48 горизонтов](data/R3_horizon.csv), [24 часа × лаг](data/R3_hour_lead.csv), [месяц × ветер](data/R3_month_wind.csv).

## 5. Рампы: направление скачка имеет значение

![Ошибки при росте и падении мощности](../../outputs/figures/R3_ramps.png)

Рампа — `|P(t)−P(t−1)| ≥ 0,3` при разнице меток **ровно один час**, отдельно для каждой турбины и лага. **{unknown} строк** без предыдущего соседнего часа выделены как неизвестные, а не спокойные. Рамповых строк **{len(ramps)} из {len(known)}** допустимых переходов (**{100 * len(ramps) / len(known):.2f}%**); факт повторяется между лагами. MAE при рампах **{ramps.abs_error.mean():.4f}**, без рампы **{t['ramps'].loc['steady', 'mae']:.4f}**.

{markdown(ramp_table, 'Режим')}

При росте bias **{t['ramps'].loc['up', 'bias']:+.4f}**, при спаде **{t['ramps'].loc['down', 'bias']:+.4f}** — прогноз сглаживает переходы. Дополнительно проверено обнаружение скачков тем же порогом 0,3: сравнивается `ΔP50` с `Δфакта` и правильным знаком. Из этой проверки исключены смены суточного выпуска на местной полуночи и все пропуски.

| Лаг | Допустимых переходов | Рамп факта | Рамп P50 | Совпали порог и знак | Полнота, % | Точность, % |
|---|---:|---:|---:|---:|---:|---:|
'''
    for lag, row in det.iterrows():
        text += f"| {lag} | {int(row.eligible_transitions)} | {int(row.actual_ramps)} | {int(row.predicted_ramps)} | {int(row.correct_direction_ramps)} | {row.directional_recall_pct:.1f} | {row.directional_precision_pct:.1f} |\n"
    text += f'''
Это строгая проверка совпадения **в тот же час**, без допуска ±1 ч; она не измеряет раннее предупреждение и не заменяет событийную оценку диспетчерской полезности.

## 6. Трудные эпизоды и возможные причины

![Худший день и соседние сутки](../../outputs/figures/R3_worst_episode.png)

Показан T1 вокруг **{day}** — дня с максимальным средним MAE обеих турбин/лагов среди дней с ≥80 строками. Это намеренно трудный, выбранный постфактум эпизод, а не типичный день. Линии имеют разрывы там, где наблюдения исключены. P50 по одному лагу складывается из суточных выпусков; нижний график показывает прогноз best_match и замер на турбине.

{markdown(worst, 'Местная дата')}

На выбранном дне T1, лаг 1: средний факт **{episode.power.mean():.3f}**, средний P50 **{episode.p50.mean():.3f}**. В час максимальной ошибки **{worst_row.local_time:%d.%m %H:%M}** факт **{worst_row.power:.3f}**, P50 **{worst_row.p50:.3f}**, прогноз ветра **{worst_row.wind_speed_100m:.2f} м/с**, замер **{worst_row.wind_meas:.2f} м/с**. Видимое расхождение ветра — возможный вклад, а не доказанная полная причина дефицита.

{markdown(t['wind_difference'], 'Абс. расхождение ветра, м/с')}

Корреляция ошибки мощности с `прогноз ветра 100 м − замер на турбине` равна **{corr:+.3f}**. Это совместимо с вкладом несоответствия ветра в ошибки мощности. Но замер и прогноз относятся к разным пространственным масштабам/высотам, а модель использует ещё ансамбль, направление и рельеф: корреляция не устанавливает единственную причину. Без журналов состояния оборудования нельзя отделить оставшиеся ограничения, ошибки SCADA и аэродинамику. Не приписываем ошибку конкретной аварии или следу турбин.

## 7. Что вынести на Demo Day

- **Сосредоточить улучшения на 3–9 м/с:** {core.row_share_pct.sum():.1f}% строк дают {core.error_share_pct.sum():.1f}% абсолютной ошибки. Проверять прогноз ветра и режимные поправки на новых временных holdout, не подгонять на этом тесте.
- **Переходы — отдельная задача:** при росте модель недодаёт, при падении завышает мощность. Нужны события, знак и время рампы, а не только общий MAE.
- **Не обещать 80% фактического покрытия:** здесь {overall.coverage_pct:.2f}%; показывать ограничения по режимам рядом с вероятностным прогнозом. Изменение калибровки — отдельная работа.
- **Не обобщать зимний бэктест на весь год:** отдельно проверять другие сезоны, оборудование и режимы ограничений; разрез по факту использовать для диагностики, а не как доступный модели признак.

## 8. Воспроизведение и проверка

```bash
PYTHONPATH=src python scripts/make_error_analysis.py
PYTHONPATH=src python -m pytest -q tests/test_error_analysis.py
```

Скрипт использует только сохранённые прогнозы и локальный погодный кэш; модель и исходные файлы бэктеста не записывает. Генерирует этот отчёт, **пять PNG** в `outputs/figures/`, таблицы `docs/research/data/R3_*.csv` и [manifest](data/R3_manifest.json) с SHA-256 входов. Проверяет ключи, временную стыковку, конечность и порядок квантилей, совпадение итоговых метрик с JSON и суммы строк в разрезах. Результаты в таблицах — полный пересчёт после R1; R2 и код ядра не менялись.

Дополнительно: [сектора направления](data/R3_sector.csv), [рампы по лагам](data/R3_ramps_lead.csv), [20 крупнейших ошибок](data/R3_largest_errors.csv). В CSV также есть RMSE, ширина, доли выходов ниже/выше интервала и 90-й перцентиль абсолютной ошибки. Столбец «Доля» рассчитан от всех {len(b)} строк, «Вклад» — от общей суммы абсолютных ошибок. Покрытие и доли выходов за интервал рассчитаны внутри каждой группы; знаменатели полноты/точности обнаружения рамп приведены в её таблице.
'''
    REPORT.write_text(text, encoding="utf-8")
    return meta


def verify_metrics(b, tables):
    metrics = json.loads((config.OUTPUTS_DIR / "backtest_metrics.json").read_text())
    for key in ["mae", "rmse", "bias"]:
        np.testing.assert_allclose(tables["overall"].iloc[0][key], metrics["overall"]["model"][key], rtol=1e-10, atol=1e-12)
    for (turbine, lead), row in tables["turbine_lead"].iterrows():
        expected = metrics["by"][f"{turbine}_lead{lead}"]
        for key in ["mae", "rmse", "bias"]:
            np.testing.assert_allclose(row[key], expected["model"][key], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(row.coverage_pct, expected["coverage_p10_p90_pct"], rtol=1e-10)
    for name, table in tables.items():
        if name != "ramp_detection":
            assert table.n.sum() == len(b), (name, table.n.sum())
            np.testing.assert_allclose(table.error_share_pct.sum(), 100)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    b = load_predictions()
    tables = make_tables(b)
    verify_metrics(b, tables)
    day = make_figures(b, tables)
    manifest = write_report(b, tables, day)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(tables["overall"].to_string())


if __name__ == "__main__":
    main()
