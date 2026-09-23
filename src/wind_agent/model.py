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

FEATURE_CONTEXT_VERSION = "exact_hour_weather_before_scada_v1"

QUANTILES = {"p10": 0.10, "p90": 0.90}
HGB_PARAMS = dict(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=40,
                  l2_regularization=1.0, random_state=42)


COVERAGE = 0.80          # номинальное покрытие интервала P10–P90
MIN_HALF_WIDTH = 0.02    # минимальная полуширина интервала при нормировке невязок (доля номинала)


@dataclass
class PowerModel:
    models: dict = field(default_factory=dict)
    features: list = field(default_factory=lambda: list(FEATURES))
    meta: dict = field(default_factory=dict)
    # конформные поправки к P10/P90 по горизонту: {lead_day: q}; интервал расширяется на q с каждой стороны
    calibration: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PowerModel":
        self.models["p50"] = HistGradientBoostingRegressor(loss="squared_error", **HGB_PARAMS).fit(X[self.features], y)
        for name, q in QUANTILES.items():
            self.models[name] = HistGradientBoostingRegressor(loss="quantile", quantile=q, **HGB_PARAMS).fit(X[self.features], y)
        return self

    def predict_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        """Квантили модели без калибровки, обрезанные в [0, 1] и упорядоченные P10 ≤ P50 ≤ P90."""
        out = pd.DataFrame({k: np.clip(m.predict(X[self.features]), 0, 1) for k, m in self.models.items()}, index=X.index)
        out["p10"] = np.minimum(out["p10"], out["p50"])
        out["p90"] = np.maximum(out["p90"], out["p50"])
        return out[["p10", "p50", "p90"]]

    @staticmethod
    def _half_widths(out: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Полуширины интервала ниже и выше P50 (не меньше MIN_HALF_WIDTH, чтобы у пола/потолка не делить на ноль)."""
        lo = np.maximum(out["p50"].to_numpy() - out["p10"].to_numpy(), MIN_HALF_WIDTH)
        hi = np.maximum(out["p90"].to_numpy() - out["p50"].to_numpy(), MIN_HALF_WIDTH)
        return lo, hi

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Квантили с конформной поправкой по горизонту: интервал расширяется пропорционально своей полуширине."""
        out = self.predict_raw(X)
        if self.calibration and "lead_day" in X:
            q = X["lead_day"].map(lambda d: self.calibration.get(int(d), 0.0)).to_numpy()
            lo, hi = self._half_widths(out)
            out["p10"] = np.clip(out["p10"].to_numpy() - q * lo, 0, 1)
            out["p90"] = np.clip(out["p90"].to_numpy() + q * hi, 0, 1)
        return out

    def calibrate(self, X: pd.DataFrame, y: pd.Series, coverage: float = COVERAGE) -> dict:
        """Split-conformal поправка для квантильной регрессии (CQR, Romano et al. 2019) с нормировкой на ширину
        интервала, отдельно по горизонту.

        Невязка s = max((P10 − y)/w_lo, (y − P90)/w_hi), где w_lo, w_hi — полуширины интервала; q — квантиль уровня
        (1−α)(1+1/n) невязок на калибровочной выборке, которую модель не видела. Интервал [P10 − q·w_lo, P90 + q·w_hi]
        покрывает факт с частотой ≈ coverage. Нормировка нужна, потому что промахи неоднородны: у пола и потолка
        мощности интервал узкий и промахи крошечные, в рабочей зоне — широкий и промахи большие.
        """
        raw = self.predict_raw(X)
        lo, hi = self._half_widths(raw)
        p10, p90, yy = raw["p10"].to_numpy(), raw["p90"].to_numpy(), np.asarray(y, dtype=float)
        leads = X["lead_day"].astype(int).to_numpy()
        self.calibration = {}
        for lead in np.unique(leads):   # позиционно: в выборке два лага на один и тот же час, метки времени дублируются
            pos = np.flatnonzero(leads == lead)
            s = np.maximum((p10[pos] - yy[pos]) / lo[pos], (yy[pos] - p90[pos]) / hi[pos])
            level = min(1.0, coverage * (1 + 1 / len(pos)))
            self.calibration[int(lead)] = float(max(0.0, np.quantile(s, level)))
        return self.calibration

    def save(self, path: Path = config.MODELS_DIR / "power_model.joblib") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "features": self.features, "meta": self.meta, "calibration": self.calibration}, path)
        return path

    @classmethod
    def load(cls, path: Path = config.MODELS_DIR / "power_model.joblib") -> "PowerModel":
        d = joblib.load(path)
        return cls(models=d["models"], features=d["features"], meta=d.get("meta", {}), calibration=d.get("calibration", {}))


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
    # калибровка интервалов: модель, обученная до CALIB_START, калибруется на CALIB_START..TRAIN_END (не видела их),
    # поправки применяются к модели на всём обучении — так же, как в train_final (калибровка на свежем holdout)
    cal = tr & (df.index >= pd.Timestamp(config.BACKTEST_CALIB_START, tz="UTC"))
    fit_only = tr & ~cal
    calib_q = PowerModel().fit(X[fit_only], y[fit_only]).calibrate(X[cal], y[cal])
    model = PowerModel().fit(X[tr], y[tr])
    raw = model.predict_raw(X[te])
    model.calibration = calib_q
    pred = model.predict(X[te])
    test = df[te].copy()
    test[["p10", "p50", "p90"]] = pred
    test["p10_raw"], test["p90_raw"] = raw["p10"], raw["p90"]
    test["power_curve"] = power_curve_baseline(df[tr], test)
    test["persistence"] = np.nan
    for t in config.TURBINES:
        m = test["turbine"] == t
        test.loc[m, "persistence"] = persistence_baseline(t, test.index[m], test.loc[m, "lead_day"]).values
    res = {"feature_context": FEATURE_CONTEXT_VERSION, "features": list(model.features), "params": HGB_PARAMS,
           "train_range": [str(df.index[tr].min()), str(df.index[tr].max())],
           "test_range": [config.BACKTEST_TEST_START, config.BACKTEST_TEST_END], "n_train": int(tr.sum()),
           "calibration": {"range": [config.BACKTEST_CALIB_START, config.BACKTEST_TRAIN_END], "n": int(cal.sum()),
                           "q_by_lead": {str(k): round(v, 4) for k, v in calib_q.items()}, "nominal_coverage": COVERAGE}, "by": {}}
    for (t, lead), g in test.groupby(["turbine", "lead_day"]):
        g2 = g.dropna(subset=["persistence"])
        res["by"][f"{t}_lead{lead}"] = {
            "model": metrics(g["power"], g["p50"]),
            "power_curve": metrics(g["power"], g["power_curve"]),
            "persistence": metrics(g2["power"], g2["persistence"]),
            "coverage_p10_p90_pct": float(100 * ((g["power"] >= g["p10"]) & (g["power"] <= g["p90"])).mean()),
            "coverage_raw_pct": float(100 * ((g["power"] >= g["p10_raw"]) & (g["power"] <= g["p90_raw"])).mean()),
            "width_p10_p90": float((g["p90"] - g["p10"]).mean()),
            "width_raw": float((g["p90_raw"] - g["p10_raw"]).mean()),
            "mean_actual": float(g["power"].mean()),
        }
    res["overall"] = {"model": metrics(test["power"], test["p50"]),
                      "power_curve": metrics(test["power"], test["power_curve"]),
                      "persistence": metrics(test.dropna(subset=["persistence"])["power"], test.dropna(subset=["persistence"])["persistence"])}
    cov = 100 * ((test["power"] >= test["p10"]) & (test["power"] <= test["p90"])).mean()
    cov_raw = 100 * ((test["power"] >= test["p10_raw"]) & (test["power"] <= test["p90_raw"])).mean()
    res["overall"]["coverage_p10_p90_pct"], res["overall"]["coverage_raw_pct"] = float(cov), float(cov_raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backtest_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    cols = ["turbine", "lead_day", "power", "p10", "p50", "p90", "p10_raw", "p90_raw", "power_curve", "persistence", "wind_speed_100m", "wind_meas"]
    test[cols].to_csv(out_dir / "backtest_predictions.csv")
    log.info("Бэктест: model MAE %.4f | power curve %.4f | persistence %.4f | покрытие P10–P90 %.1f%% (без калибровки %.1f%%)",
             res["overall"]["model"]["mae"], res["overall"]["power_curve"]["mae"], res["overall"]["persistence"]["mae"], cov, cov_raw)
    return res


def train_final(wx: WeatherClient) -> PowerModel:
    """Финальная модель на всей истории до 31.01.2026 (для тестового февраля и live).

    Конформные поправки интервала оцениваются на последних двух месяцах с фактом (дек.2025–янв.2026)
    моделью, обученной без них, и применяются к модели на всей истории.
    """
    df, X, y = build_dataset(wx)
    cal = df.index >= pd.Timestamp(config.BACKTEST_TEST_START, tz="UTC")
    calib_q = PowerModel().fit(X[~cal], y[~cal]).calibrate(X[cal], y[cal])
    model = PowerModel().fit(X, y)
    model.calibration = calib_q
    model.meta = {"train_start": str(df.index.min()), "train_end": str(df.index.max()), "n_train": int(len(df)),
                  "turbines": list(config.TURBINES), "features": FEATURES, "params": HGB_PARAMS,
                  "feature_context": FEATURE_CONTEXT_VERSION,
                  "calibration": {"range": [config.BACKTEST_TEST_START, config.HISTORY_END], "n": int(cal.sum()),
                                  "q_by_lead": {str(k): round(v, 4) for k, v in calib_q.items()}, "nominal_coverage": COVERAGE}}
    path = model.save()
    log.info("Модель обучена на %d строках (%s..%s), поправки интервала %s → %s", len(df), df.index.min(), df.index.max(),
             {k: round(v, 3) for k, v in calib_q.items()}, path)
    return model
