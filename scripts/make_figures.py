"""Графики бэктеста для README: факт vs прогноз (P10–P90), ошибка по лагу, кривая мощности."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from wind_agent import config
from wind_agent.data import utc_to_local, load_raw

OUT = config.OUTPUTS_DIR / "figures"; OUT.mkdir(parents=True, exist_ok=True)
bt = pd.read_csv(config.OUTPUTS_DIR / "backtest_predictions.csv", parse_dates=["time"])
bt["time"] = pd.to_datetime(bt["time"], utc=True)
bt["local"] = utc_to_local(pd.DatetimeIndex(bt["time"]))
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

# 1. Две недели января 2026, турбина 1, горизонт 24 ч и 48 ч
fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
for ax, lead in zip(axes, (1, 2)):
    g = bt[(bt.turbine == "t1") & (bt.lead_day == lead) & (bt.local >= "2026-01-10") & (bt.local < "2026-01-24")].sort_values("local")
    ax.fill_between(g.local, g.p10, g.p90, color="tab:blue", alpha=0.18, label="P10–P90")
    ax.plot(g.local, g.p50, color="tab:blue", lw=1.4, label="прогноз P50")
    ax.plot(g.local, g.power, color="black", lw=1.0, label="факт")
    ax.plot(g.local, g.persistence, color="tab:gray", lw=0.8, ls="--", label="персистентность")
    mae = np.abs(g.p50 - g.power).mean(); mae_p = np.abs(g.persistence - g.power).mean()
    ax.set_title(f"Турбина 1, горизонт {'1–24' if lead == 1 else '25–48'} ч (прогноз погоды за {lead} сут.): MAE модели {mae:.3f}, персистентности {mae_p:.3f}")
    ax.set_ylabel("мощность, доля номинала"); ax.set_ylim(-0.02, 1.05); ax.legend(loc="upper right", ncol=4, fontsize=8)
axes[-1].set_xlabel("местное время (UTC+5)")
fig.tight_layout(); fig.savefig(OUT / "backtest_t1_jan2026.png", dpi=130); plt.close(fig)

# 2. MAE по турбинам, лагам и методам
m = json.load(open(config.OUTPUTS_DIR / "backtest_metrics.json"))
keys = ["t1_lead1", "t1_lead2", "t2_lead1", "t2_lead2"]
labels = ["T1, 1–24 ч", "T1, 25–48 ч", "T2, 1–24 ч", "T2, 25–48 ч"]
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(len(keys)); w = 0.26
for i, (meth, name, col) in enumerate([("persistence", "персистентность", "tab:gray"), ("power_curve", "кривая мощности", "tab:orange"), ("model", "модель GBM", "tab:blue")]):
    vals = [m["by"][k][meth]["mae"] for k in keys]
    ax.bar(x + (i - 1) * w, vals, w, label=name, color=col)
    for xi, v in zip(x + (i - 1) * w, vals):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("MAE, доля номинала"); ax.set_title("Бэктест 01.12.2025–31.01.2026 (архивные прогнозы Open-Meteo с честным лагом)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "backtest_mae_by_lead.png", dpi=130); plt.close(fig)

# 3. Кривая мощности по 10-мин данным + по прогнозному ветру
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
raw = load_raw("t1").sample(30000, random_state=1)
axes[0].scatter(raw.wind_meas, raw.power, s=2, alpha=0.25, color="tab:blue")
axes[0].set_title("Замер на турбине 1: ветер → мощность (10-мин)"); axes[0].set_xlabel("скорость ветра на турбине, м/с"); axes[0].set_ylabel("мощность, доля номинала")
g = bt[bt.turbine == "t1"]
axes[1].scatter(g.wind_speed_100m, g.power, s=3, alpha=0.25, color="tab:orange", label="факт")
bins = np.arange(0, 26, 1.0); cur = g.groupby(pd.cut(g.wind_speed_100m, bins), observed=False).agg(power=("power", "mean"), p50=("p50", "mean"))
mid = bins[:-1] + 0.5
axes[1].plot(mid, cur.power.values, color="black", lw=1.5, label="средний факт по бинам"); axes[1].plot(mid, cur.p50.values, color="tab:blue", lw=1.5, ls="--", label="средний прогноз P50")
axes[1].set_title("Прогноз ветра 100 м (лаг 1–2 сут.) → мощность, тест"); axes[1].set_xlabel("прогноз ветра на 100 м, м/с"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "power_curve.png", dpi=130); plt.close(fig)
print("saved", sorted(p.name for p in OUT.glob("*.png")))
