"""R3: анализ ошибок бэктеста по режимам — где и почему модель ошибается.

Вход: outputs/backtest_predictions.csv (после wind-agent backtest) + прогноз направления из кэша.
Выход: outputs/figures/error_analysis.png и docs/research/R3_error_analysis.md с таблицами.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_agent import config
from wind_agent.data import utc_to_local
from wind_agent.weather import WeatherClient

OUT_FIG = config.OUTPUTS_DIR / "figures" / "error_analysis.png"
OUT_MD = config.ROOT / "docs" / "research" / "R3_error_analysis.md"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

bt = pd.read_csv(config.OUTPUTS_DIR / "backtest_predictions.csv", parse_dates=["time"])
bt["time"] = pd.to_datetime(bt["time"], utc=True)
bt["err"] = bt["p50"] - bt["power"]
bt["abs_err"] = bt["err"].abs()
bt["hour"] = utc_to_local(pd.DatetimeIndex(bt["time"])).hour
bt["inside"] = (bt["power"] >= bt["p10"]) & (bt["power"] <= bt["p90"])

# направление прогноза (best_match, лаг как в строке) из кэша Previous Runs
wx = WeatherClient(offline=True)
dirs = []
for t in config.TURBINES:
    raw = wx.previous_runs(t, "2025-12-01", "2026-01-31").set_index("time")
    for lead in config.LEAD_DAYS:
        d = raw[f"wind_direction_100m_previous_day{lead}"].rename("dir").reset_index()
        d["turbine"], d["lead_day"] = t, lead
        dirs.append(d)
bt = bt.merge(pd.concat(dirs), on=["time", "turbine", "lead_day"], how="left")

# режимы
ws_bins = [0, 3, 6, 9, 12, 40]
ws_labels = ["штиль <3", "разгон 3–6", "рабочая 6–9", "выход на номинал 9–12", "номинал >12"]
bt["ws_regime"] = pd.cut(bt["wind_speed_100m"], ws_bins, labels=ws_labels, include_lowest=True)
pw_bins = [-0.001, 0.05, 0.3, 0.7, 0.95, 1.0]
pw_labels = ["факт ≤0.05", "0.05–0.3", "0.3–0.7", "0.7–0.95", ">0.95"]
bt["pw_regime"] = pd.cut(bt["power"], pw_bins, labels=pw_labels)
sector_names = ["С", "ССВ", "СВ", "ВСВ", "В", "ВЮВ", "ЮВ", "ЮЮВ", "Ю", "ЮЮЗ", "ЮЗ", "ЗЮЗ", "З", "ЗСЗ", "СЗ", "ССЗ"]
bt["sector"] = ((bt["dir"] + 11.25) // 22.5 % 16).astype("Int64")
# рампы факта: |Δ мощности| за час ≥ 0.3
bt = bt.sort_values(["turbine", "lead_day", "time"])
bt["ramp"] = bt.groupby(["turbine", "lead_day"])["power"].diff().abs() >= 0.3


def table(by: str, order=None) -> pd.DataFrame:
    g = bt.groupby(by, observed=True).agg(hours=("err", "size"), mae=("abs_err", "mean"), bias=("err", "mean"),
                                          coverage=("inside", "mean"), mean_actual=("power", "mean"))
    g["share_%"] = 100 * g["hours"] / len(bt)
    g["coverage"] *= 100
    return g.reindex(order) if order is not None else g


t_ws = table("ws_regime", ws_labels)
t_pw = table("pw_regime", pw_labels)
t_hour = table("hour")
t_lead = table("lead_day")
t_sector = table("sector"); t_sector.index = [sector_names[int(i)] for i in t_sector.index]
t_ramp = table("ramp"); t_ramp.index = ["обычный час", "рампа факта ≥0.3/ч"]

fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
ax = axes[0, 0]; t_ws[["mae"]].plot.bar(ax=ax, color="tab:blue", legend=False); ax.set_title("MAE по прогнозной скорости ветра (100 м)"); ax.set_xlabel(""); ax.set_ylabel("MAE")
for i, (m, b) in enumerate(zip(t_ws["mae"], t_ws["bias"])):
    ax.text(i, m + 0.005, f"bias {b:+.2f}", ha="center", fontsize=7)
ax = axes[0, 1]; t_pw[["mae"]].plot.bar(ax=ax, color="tab:orange", legend=False); ax.set_title("MAE по фактическому уровню мощности"); ax.set_xlabel(""); ax.set_ylabel("MAE")
for i, (m, b) in enumerate(zip(t_pw["mae"], t_pw["bias"])):
    ax.text(i, m + 0.005, f"bias {b:+.2f}", ha="center", fontsize=7)
ax = axes[0, 2]; ax.plot(t_hour.index, t_hour["mae"], marker="o", ms=3, label="MAE"); ax.plot(t_hour.index, t_hour["bias"], marker="s", ms=3, color="tab:red", label="bias"); ax.axhline(0, color="black", lw=0.5)
ax.set_title("Ошибка по часу суток (местное время)"); ax.set_xlabel("час"); ax.legend(fontsize=8); ax.set_xticks(range(0, 24, 3))
ax = axes[1, 0]; big = t_sector[t_sector["hours"] >= 100]; big[["mae"]].plot.bar(ax=ax, color="tab:green", legend=False); ax.set_title("MAE по сектору направления (секторы ≥100 ч)"); ax.set_xlabel("")
for i, (m, b) in enumerate(zip(big["mae"], big["bias"])):
    ax.text(i, m + 0.005, f"{b:+.2f}", ha="center", fontsize=7)
ax = axes[1, 1]; ax.hist(bt["err"], bins=60, color="tab:gray"); ax.set_title(f"Распределение ошибки P50 − факт (bias {bt['err'].mean():+.3f}, MAE {bt['abs_err'].mean():.3f})"); ax.set_xlabel("ошибка, доля номинала")
ax = axes[1, 2]
for lead, g in bt.groupby("lead_day"):
    q = g.groupby(pd.cut(g["power"], np.linspace(0, 1, 11)), observed=True)["inside"].mean() * 100
    ax.plot(np.linspace(0.05, 0.95, 10), q.values, marker="o", ms=3, label=f"горизонт {'1–24' if lead == 1 else '25–48'} ч")
ax.axhline(80, color="tab:red", ls=":", lw=0.8); ax.set_title("Покрытие P10–P90 по уровню факта (номинал 80 %)"); ax.set_xlabel("фактическая мощность"); ax.set_ylabel("%"); ax.legend(fontsize=8)
fig.suptitle("Анализ ошибок бэктеста 01.12.2025–31.01.2026 (обе турбины, горизонты 1–48 ч, прогноз с честным лагом)", fontsize=11)
fig.tight_layout(); OUT_FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT_FIG, dpi=130); plt.close(fig)


def md(df: pd.DataFrame, name: str) -> str:
    d = df.copy(); d.index.name = name
    d = d[["hours", "share_%", "mean_actual", "mae", "bias", "coverage"]].round(3)
    d.columns = ["часов", "доля, %", "средний факт", "MAE", "bias", "покрытие P10–P90, %"]
    lines = ["| " + name + " | " + " | ".join(d.columns) + " |", "|---" * (len(d.columns) + 1) + "|"]
    for idx, row in d.iterrows():
        lines.append("| " + str(idx) + " | " + " | ".join(f"{v:+.3f}" if c == "bias" else (f"{v:.0f}" if c in ("часов",) else f"{v:.1f}" if c in ("доля, %", "покрытие P10–P90, %") else f"{v:.3f}") for c, v in row.items()) + " |")
    return "\n".join(lines)


OUT_MD.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.write_text(f"""# R3. Анализ ошибок бэктеста по режимам

Данные: `outputs/backtest_predictions.csv` — тест 01.12.2025–31.01.2026, обе турбины, горизонты 1–48 ч ({len(bt)} часов),
модель с калибровкой интервалов (поправки окт–ноя). Ошибка = P50 − факт, доля номинала. Скрипт: `scripts/make_error_analysis.py`.

![Анализ ошибок](../../outputs/figures/error_analysis.png)

## По прогнозной скорости ветра на 100 м

{md(t_ws, "режим")}

## По фактическому уровню мощности

{md(t_pw, "факт")}

## По горизонту

{md(t_lead, "lead_day")}

## По сектору направления прогноза (секторы с ≥ 100 ч)

{md(t_sector[t_sector["hours"] >= 100], "сектор")}

## Рампы

{md(t_ramp, "режим")}

## Выводы

- Самые большие ошибки — в рабочей зоне кривой мощности (прогноз 6–12 м/с, факт 0.3–0.95): там крутая кривая, и ошибка ветра в 2 м/с превращается в 0.3–0.4 номинала. При штиле и на номинале ошибки малы.
- Смещение по уровню факта: при факте ≤ 0.05 модель завышает (предсказывает «ожидаемую» мощность при неточном прогнозе слабого ветра), при факте > 0.95 занижает — это цена усреднения при регрессии на зашумлённый прогноз, а не дефект модели.
- Часы факта с рампой ≥ 0.3/ч — самые дорогие для диспетчера и самые трудные для модели: см. таблицу «Рампы».
- Покрытие интервала минимально в средней зоне мощности и максимально у пола/потолка — интервалы имеет смысл расширять по режиму (см. исследование R2).
""", encoding="utf-8")
print("saved:", OUT_FIG.name, OUT_MD.relative_to(config.ROOT))
