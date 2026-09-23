"""Модель прогноза мощности: градиентный бустинг (sklearn HistGradientBoosting) для P50 и квантилей P10/P90."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config
from .data import load_hourly, training_frame
from .features import FEATURES, make_features
from .weather import WeatherClient

log = logging.getLogger(__name__)

QUANTILES = {"p10": 0.10, "p90": 0.90}
HGB_PARAMS = dict(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=40,
                  l2_regularization=1.0, random_state=42)


@dataclass
class PowerModel:
    models: dict = field(default_factory=dict)
    features: list = field(default_factory=lambda: list(FEATURES))
    meta: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PowerModel":
        self.models["p50"] = HistGradientBoostingRegressor(loss="squared_error", **HGB_PARAMS).fit(X[self.features], y)
        for name, q in QUANTILES.items():
            self.models[name] = HistGradientBoostingRegressor(loss="quantile", quantile=q, **HGB_PARAMS).fit(X[self.features], y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({k: np.clip(m.predict(X[self.features]), 0, 1) for k, m in self.models.items()}, index=X.index)
        # квантили не должны пересекаться
        out["p10"] = np.minimum(out["p10"], out["p50"])
        out["p90"] = np.maximum(out["p90"], out["p50"])
        return out[["p10", "p50", "p90"]]

    def save(self, path: Path = config.MODELS_DIR / "power_model.joblib") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "features": self.features, "meta": self.meta}, path)
        return path

    @classmethod
    def load(cls, path: Path = config.MODELS_DIR / "power_model.joblib") -> "PowerModel":
        d = joblib.load(path)
        return cls(models=d["models"], features=d["features"], meta=d.get("meta", {}))


def build_dataset(wx: WeatherClient, end: str = config.HISTORY_END) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Обучающая выборка по обеим турбинам: (кадр с метаданными, X, y)."""
    frames = [training_frame(t, wx, end=end) for t in config.TURBINES]
    df = pd.concat(frames).sort_index(kind="stable")
    X = make_features(df)
    y = df["power"].astype(float)
    return df, X, y


def metrics(y: pd.Series, p: pd.Series) -> dict:
    e = p - y
    return {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()), "n_mae_pct": float(100 * e.abs().mean()),  # мощность нормирована на номинал → nMAE = MAE·100 %
            "n": int(len(y))}


def persistence_baseline(turbine: str, index_utc: pd.DatetimeIndex, lead_day: pd.Series) -> pd.Series:
    """Персистентность: мощность «сегодня в тот же час» (известна на момент выпуска в 23:00 дня D)."""
    scada = load_hourly(turbine)["power"]
    shifted = index_utc - pd.to_timedelta(lead_day.values * 24, unit="h")
    return pd.Series(scada.reindex(shifted).values, index=index_utc)


def power_curve_baseline(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Кривая мощности по прогнозному ветру 100 м: средняя мощность в бинах 0.5 м/с на обучении."""
    bins = np.arange(0, 45, 0.5)
    curve = train["power"].groupby(pd.cut(train["wind_speed_100m"], bins), observed=False).mean()
    pred = pd.cut(test["wind_speed_100m"], bins).map(curve).astype(float)
    return pred.fillna(train["power"].mean())


def backtest(wx: WeatherClient, out_dir: Path = config.OUTPUTS_DIR) -> dict:
    """Честный бэктест по протоколу ТЗ на периоде с эталоном: обучение до BACKTEST_TRAIN_END, тест после."""
    df, X, y = build_dataset(wx)
    tr = df.index <= pd.Timestamp(config.BACKTEST_TRAIN_END, tz="UTC") + pd.Timedelta(hours=23)
    te = (df.index >= pd.Timestamp(config.BACKTEST_TEST_START, tz="UTC")) & (df.index <= pd.Timestamp(config.BACKTEST_TEST_END, tz="UTC") + pd.Timedelta(hours=23))
    model = PowerModel().fit(X[tr], y[tr])
    pred = model.predict(X[te])
    test = df[te].copy()
    test[["p10", "p50", "p90"]] = pred
    test["power_curve"] = power_curve_baseline(df[tr], test)
    test["persistence"] = np.nan
    for t in config.TURBINES:
        m = test["turbine"] == t
        test.loc[m, "persistence"] = persistence_baseline(t, test.index[m], test.loc[m, "lead_day"]).values
    res = {"train_range": [str(df.index[tr].min()), str(df.index[tr].max())],
           "test_range": [config.BACKTEST_TEST_START, config.BACKTEST_TEST_END], "n_train": int(tr.sum()), "by": {}}
    for (t, lead), g in test.groupby(["turbine", "lead_day"]):
        g2 = g.dropna(subset=["persistence"])
        res["by"][f"{t}_lead{lead}"] = {
            "model": metrics(g["power"], g["p50"]),
            "power_curve": metrics(g["power"], g["power_curve"]),
            "persistence": metrics(g2["power"], g2["persistence"]),
            "coverage_p10_p90_pct": float(100 * ((g["power"] >= g["p10"]) & (g["power"] <= g["p90"])).mean()),
            "mean_actual": float(g["power"].mean()),
        }
    res["overall"] = {"model": metrics(test["power"], test["p50"]),
                      "power_curve": metrics(test["power"], test["power_curve"]),
                      "persistence": metrics(test.dropna(subset=["persistence"])["power"], test.dropna(subset=["persistence"])["persistence"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backtest_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    cols = ["turbine", "lead_day", "power", "p10", "p50", "p90", "power_curve", "persistence", "wind_speed_100m", "wind_meas"]
    test[cols].to_csv(out_dir / "backtest_predictions.csv")
    log.info("Бэктест: model MAE %.4f | power curve %.4f | persistence %.4f",
             res["overall"]["model"]["mae"], res["overall"]["power_curve"]["mae"], res["overall"]["persistence"]["mae"])
    return res


def train_final(wx: WeatherClient) -> PowerModel:
    """Финальная модель на всей истории до 31.01.2026 (для тестового февраля и live)."""
    df, X, y = build_dataset(wx)
    model = PowerModel().fit(X, y)
    model.meta = {"train_start": str(df.index.min()), "train_end": str(df.index.max()), "n_train": int(len(df)),
                  "turbines": list(config.TURBINES), "features": FEATURES, "params": HGB_PARAMS}
    path = model.save()
    log.info("Модель обучена на %d строках (%s..%s) → %s", len(df), df.index.min(), df.index.max(), path)
    return model
