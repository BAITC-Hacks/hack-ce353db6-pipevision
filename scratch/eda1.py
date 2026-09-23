import pandas as pd, numpy as np

BASE = "data/raw/"
COLS = ["id", "time", "wind", "power", "temp"]
d = {}; HH = {}
for k in ["1", "2"]:
    df = pd.read_csv(BASE + f"turbine_{k}.csv"); df.columns = COLS
    df["time"] = pd.to_datetime(df["time"]); d[k] = df.set_index("time")

for k, df in d.items():
    print(f"\n######## turbine {k}")
    # power curve: binned by wind speed
    bins = np.arange(0, 24, 1.0)
    g = df.groupby(pd.cut(df["wind"], bins))["power"].agg(["mean", "median", "std", "count"]).round(3)
    print("power curve by wind bin:\n", g.to_string())
    # zeros with high wind (curtailment / outage)
    z = df[(df["power"] == 0) & (df["wind"] > 5)]
    print("power==0 & wind>5:", len(z), "| power<0.02 & wind>8:", ((df['power'] < 0.02) & (df['wind'] > 8)).sum())
    # hourly aggregation
    h = df[["wind", "power", "temp"]].resample("1h").mean()
    cnt = df["power"].resample("1h").count()
    print("hourly rows:", len(h), "| hours with <6 samples:", (cnt.between(1, 5)).sum(), "| empty hours:", (cnt == 0).sum())
    # diurnal & seasonal
    hh = h.dropna()
    print("mean power by hour of day:", hh.groupby(hh.index.hour)["power"].mean().round(3).to_dict())
    print("mean power by month:", hh.groupby(hh.index.month)["power"].mean().round(3).to_dict())
    print("mean wind by month:", hh.groupby(hh.index.month)["wind"].mean().round(2).to_dict())
    print("mean power by year:", hh.groupby(hh.index.year)["power"].mean().round(3).to_dict())
    # persistence baselines on hourly power (24h and 48h ahead)
    for lag in [1, 24, 48]:
        e = (hh["power"] - hh["power"].shift(lag)).abs()
        print(f"persistence MAE lag {lag}h: {e.mean():.4f}")
    # autocorrelation
    print("autocorr power 1h/6h/24h/48h:", [round(hh["power"].autocorr(l), 3) for l in [1, 6, 24, 48]])
    # temp vs power/wind
    print("corr(wind,power)=%.3f corr(temp,power)=%.3f corr(temp,wind)=%.3f" % (
        hh["wind"].corr(hh["power"]), hh["temp"].corr(hh["power"]), hh["temp"].corr(hh["wind"])))
    HH[k] = h

# turbine cross-correlation
j = HH["1"].join(HH["2"], lsuffix="_1", rsuffix="_2").dropna()
print("\n#### cross-turbine (hourly): corr wind %.3f corr power %.3f, MAE power diff %.4f" % (
    j["wind_1"].corr(j["wind_2"]), j["power_1"].corr(j["power_2"]), (j["power_1"] - j["power_2"]).abs().mean()))
print("Jan 2026 power mean t1 %.3f t2 %.3f" % (j.loc["2026-01", "power_1"].mean(), j.loc["2026-01", "power_2"].mean()))
print("Feb power mean by year t1:", {y: round(HH['1'].loc[f'{y}-02', 'power'].mean(), 3) for y in [2024, 2025]})
HH["1"].to_csv(BASE + "../turbine_1_hourly.csv"); HH["2"].to_csv(BASE + "../turbine_2_hourly.csv")
