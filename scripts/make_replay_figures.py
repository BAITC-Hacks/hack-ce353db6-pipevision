"""Графики прогона тестового периода (без факта): выработка по выпускам, ревизии, ход прогноза на весь февраль.

Запуск: python scripts/make_replay_figures.py  (после wind-agent replay) → outputs/figures/replay_*.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_agent import config

OUT = config.OUTPUTS_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

summary = pd.read_csv(config.OUTPUTS_DIR / "replay_summary.csv", parse_dates=["issue_date"]).sort_values("issue_date")
latest = pd.read_csv(config.OUTPUTS_DIR / "forecasts" / "latest_by_target.csv")
latest["t"] = pd.to_datetime(latest["target_time_local"].str.slice(0, 19))
farm = latest[latest["turbine"] == "farm"].sort_values("t")

# 1. Сшитый прогноз парка на весь тестовый период: последний выпуск на каждый час, интервал P10–P90, ревизии
fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, height_ratios=[3, 1.2])
ax = axes[0]
ax.fill_between(farm["t"], farm["p10"], farm["p90"], color="tab:blue", alpha=0.18, label="P10–P90")
ax.plot(farm["t"], farm["p50"], color="tab:blue", lw=1.3, label="P50, последний выпуск на каждый час (горизонт 1–24 ч)")
prev = farm.dropna(subset=["previous_p50"])
ax.plot(prev["t"], prev["previous_p50"], color="tab:orange", lw=0.9, ls="--", label="P50 предыдущего выпуска (горизонт 25–48 ч)")
ax.set_ylim(-0.02, 1.05); ax.set_ylabel("мощность парка, доля номинала")
ax.set_title("Тестовый период 01.02–01.03.2026: почасовой прогноз парка по 28 ежедневным выпускам (факта за февраль в данных нет)")
ax.legend(loc="upper right", ncol=3, fontsize=8)
ax = axes[1]
rev = farm.dropna(subset=["revision"])
ax.bar(rev["t"], rev["revision"], width=pd.Timedelta(hours=1), color=np.where(rev["revision"] >= 0, "tab:green", "tab:red"), alpha=0.7)
ax.axhline(0, color="black", lw=0.6)
ax.set_ylabel("ревизия P50\n(новый − старый)"); ax.set_xlabel("местное время (UTC+5)")
fig.tight_layout(); fig.savefig(OUT / "replay_feb2026_farm.png", dpi=130); plt.close(fig)

# 2. Энергия по выпускам (часы номинала за 48 ч, D+1 / D+2) и MAE ревизии с флагами
fig, ax1 = plt.subplots(figsize=(13, 4))
x = np.arange(len(summary)); w = 0.38
ax1.bar(x - w / 2, summary["farm_energy_day1"], w, color="tab:blue", label="сутки D+1 (1–24 ч)")
ax1.bar(x + w / 2, summary["farm_energy_day2"], w, color="tab:cyan", label="сутки D+2 (25–48 ч)")
ax1.set_ylabel("часов номинала, парк"); ax1.set_xticks(x); ax1.set_xticklabels(summary["issue_date"].dt.strftime("%d.%m"), rotation=0)
ax1.set_xlabel("дата выпуска (23:00 местного)")
ax2 = ax1.twinx()
ax2.plot(x, summary["revision_mae"], color="tab:red", marker="o", ms=3, lw=1, label="MAE ревизии к предыдущему выпуску")
ax2.axhline(0.15, color="tab:red", ls=":", lw=0.8); ax2.set_ylabel("MAE ревизии, доля номинала"); ax2.set_ylim(0, max(0.3, summary["revision_mae"].max() * 1.15))
ax2.grid(False)
for i, (st, dec, fl) in enumerate(zip(summary["status"], summary["decision"], summary["flags"].fillna(""))):
    if st == "warning":
        ax1.text(i, summary["farm_energy_48h"].max() * 0.52, dec, rotation=90, ha="center", va="bottom", fontsize=7, color="tab:red")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=3)
n_llm = int(summary["llm_used"].sum())
ax1.set_title(f"28 выпусков: выработка парка по суткам и ревизии; решения агента (LLM в {n_llm} выпусках), порог существенной ревизии 0,15")
fig.tight_layout(); fig.savefig(OUT / "replay_feb2026_issues.png", dpi=130); plt.close(fig)
print("saved:", sorted(p.name for p in OUT.glob("replay_*.png")))
