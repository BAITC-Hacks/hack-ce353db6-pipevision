import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor

BASE = "data/"
COLS = ["id", "time", "wind", "power", "temp"]
H = {}
for k in ["1", "2"]:
    df = pd.read_csv(BASE + f"raw/turbine_{k}.csv"); df.columns = COLS
    df["time"] = pd.to_datetime(df["time"]); df = df.set_index("time")
    h = df[["wind", "power", "temp"]].resample("1h").mean()
    h["n"] = df["power"].resample("1h").count()
    H[k] = h
    h.to_csv(BASE + f"turbine_{k}_hourly_localtime.csv")

# ---- turbine 2 summary + cross-turbine
h2 = H["2"].dropna()
print("T2 mean power by month:", h2.groupby(h2.index.month)["power"].mean().round(3).to_dict())
print("T2 corr(wind,power)=%.3f; persistence MAE 24h=%.4f" % (h2["wind"].corr(h2["power"]), (h2["power"] - h2["power"].shift(24)).abs().mean()))
j = H["1"].join(H["2"], lsuffix="_1", rsuffix="_2").dropna()
print("cross-turbine: corr wind %.3f, corr power %.3f, MAE(p1-p2) %.4f" % (
    j["wind_1"].corr(j["wind_2"]), j["power_1"].corr(j["power_2"]), (j["power_1"] - j["power_2"]).abs().mean()))

# ---- timezone alignment vs Open-Meteo (UTC) historical forecast, wind 100m & 10m
om = pd.read_csv(BASE + "cache/openmeteo/openmeteo_histforecast_t1_utc.csv", parse_dates=["time"]).set_index("time")
era = pd.read_csv(BASE + "cache/openmeteo/openmeteo_era5_t1_utc.csv", parse_dates=["time"]).set_index("time")
h1 = H["1"]
for label, (a, b) in {"2023-06..2023-12": ("2023-06-01", "2023-12-31"), "2024-06..2024-12": ("2024-06-01", "2024-12-31"),
                      "2025-06..2025-12": ("2025-06-01", "2025-12-31"), "2026-01": ("2026-01-01", "2026-01-31")}.items():
    seg = h1.loc[a:b, "wind"].dropna()
    res = {}
    for sh in range(-8, 9):  # local = utc + sh  -> utc index shifted by +sh
        x = om["wind_speed_100m"].copy(); x.index = x.index + pd.Timedelta(hours=sh)
        c = seg.corr(x.reindex(seg.index))
        res[sh] = round(c, 3)
    best = max(res, key=res.get)
    print(f"tz align {label}: best shift +{best}h corr={res[best]} | {res}")
for sh in (5, 6):
    x = era["wind_speed_100m"].copy(); x.index = x.index + pd.Timedelta(hours=sh)
    seg = h1.loc["2025-10-01":"2026-01-31", "wind"].dropna()
    print(f"ERA5 check shift +{sh}: corr={seg.corr(x.reindex(seg.index)):.3f}")

# ---- assume local = UTC+5 after 2024-03-01 and UTC+6 before (verify above); build utc index
def to_utc(idx):
    off = np.where(idx < pd.Timestamp("2024-03-01"), 6, 5)
    return idx - pd.to_timedelta(off, unit="h")

# ---- forecast skill: measured wind vs forecast wind (histforecast = ~day0, prevruns day1/day2)
pr = pd.read_csv(BASE + "cache/openmeteo/openmeteo_prevruns_t1_utc.csv", parse_dates=["time"]).set_index("time")
m = h1.loc["2025-11-01":"2026-01-31"].copy(); m.index = to_utc(m.index); m = m.dropna()
for col in ["wind_speed_100m", "wind_speed_100m_previous_day1", "wind_speed_100m_previous_day2", "wind_speed_10m_previous_day1", "wind_speed_80m_previous_day1", "wind_speed_120m_previous_day1"]:
    x = pr[col].reindex(m.index)
    print(f"skill {col}: corr={m['wind'].corr(x):.3f}  MAE={np.abs(m['wind']-x).mean():.2f}  bias(fc-meas)={(x-m['wind']).mean():+.2f}  ratio meas/fc={m['wind'].mean()/x.mean():.2f}")

# ---- quick GBM: histforecast features -> hourly power, time split
feat = om.copy()
feat["hour"] = feat.index.hour; feat["doy"] = feat.index.dayofyear; feat["month"] = feat.index.month
for k in ["1", "2"]:
    y = H[k]["power"].copy(); y.index = to_utc(y.index); y = y[H[k]["n"].values >= 4].dropna()
    X = feat.reindex(y.index).dropna(); y = y.loc[X.index]
    tr = X.index < "2025-11-01"; te = ~tr
    mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0)
    mdl.fit(X[tr], y[tr]); p = np.clip(mdl.predict(X[te]), 0, 1)
    mae = np.abs(p - y[te]).mean(); rmse = np.sqrt(((p - y[te]) ** 2).mean())
    # power-curve-only baseline: bin mean of power by forecast wind100
    bins = pd.cut(X["wind_speed_100m"], np.arange(0, 40, 0.5)); pc = y[tr].groupby(bins[tr]).mean()
    pb = bins[te].map(pc).astype(float).fillna(y[tr].mean())
    print(f"GBM T{k} (train<2025-11, test Nov25-Jan26, day0 forecast feats): MAE={mae:.4f} RMSE={rmse:.4f} | powercurve baseline MAE={np.abs(pb-y[te]).mean():.4f} | mean y test={y[te].mean():.3f}")
    # same GBM evaluated with previous_day1 / day2 forecasts as inputs (true lead time)
    for d in (1, 2):
        Xd = X[te].copy()
        for v in ["wind_speed_10m", "wind_speed_80m", "wind_speed_100m", "wind_speed_120m", "wind_direction_100m", "temperature_2m"]:
            Xd[v] = pr[f"{v}_previous_day{d}"].reindex(Xd.index)
        Xd = Xd.dropna(subset=["wind_speed_100m"]); pd_ = np.clip(mdl.predict(Xd), 0, 1)
        print(f"   -> with previous_day{d} inputs: MAE={np.abs(pd_ - y[Xd.index]).mean():.4f} (n={len(Xd)})")
