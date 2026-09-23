"""R2 research candidate: CQR prototype; NOT enabled in the production model.

Coverage target 78–82% was not met. See docs/research/R2_calibration.md.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from wind_agent import config
from wind_agent.model import PowerModel as BasePowerModel, HGB_PARAMS, QUANTILES

log = logging.getLogger(__name__)

CALIBRATION_FRACTION = 0.2
CALIBRATION_SEED = 42
CALIBRATION_WIND_BINS = (4.0, 8.0, 12.0)
MIN_CALIBRATION_GROUP = 50


def conformal_correction(scores: np.ndarray, alpha: float = 0.2) -> float:
    """Порядковая статистика ceil((n+1)*(1-alpha)), без интерполяции.

    При недостаточном n возвращаем 1: для мощности в [0, 1] это полный интервал.
    Отрицательная поправка CQR допустима и сужает избыточно широкий интервал.
    """
    scores = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1 or scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Нужны непустые конечные scores и 0 < alpha < 1")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return 1.0 if rank > len(scores) else float(np.partition(scores, rank - 1)[rank - 1])


@dataclass
class PowerModel(BasePowerModel):
    calibration: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series, *, calibrate_intervals: bool = False) -> "PowerModel":
        """P50 учится на всей истории; квантили — без отложенных калибровочных дней.

        calibrate_intervals=False сохраняет исходную модель для сравнения. Для CQR
        все турбины и оба лага одного часа остаются по одну сторону разделения.
        """
        if not X.index.equals(y.index):
            raise ValueError("Индексы X и y должны совпадать, включая порядок повторов")
        fit = np.ones(len(X), dtype=bool)
        if calibrate_intervals:
            if not isinstance(X.index, pd.DatetimeIndex) or X.index.hasnans:
                raise ValueError("Временная калибровка требует DatetimeIndex без NaT")
            if X.index.tz is None:
                raise ValueError("Индекс калибровки должен иметь часовой пояс")
            dates = X.index.tz_convert("UTC").normalize()
            unique = dates.unique().sort_values()
            rng = np.random.default_rng(CALIBRATION_SEED)
            chosen = rng.choice(unique, size=math.ceil(len(unique) * CALIBRATION_FRACTION), replace=False)
            fit = ~dates.isin(chosen)
            if fit.sum() < MIN_CALIBRATION_GROUP or (~fit).sum() < MIN_CALIBRATION_GROUP:
                raise ValueError("Недостаточно истории для обучения и отдельной калибровки")
        self.calibration = {}
        self.models["p50"] = HistGradientBoostingRegressor(loss="squared_error", **HGB_PARAMS).fit(X[self.features], y)
        for name, q in QUANTILES.items():
            self.models[name] = HistGradientBoostingRegressor(loss="quantile", quantile=q, **HGB_PARAMS).fit(X.loc[fit, self.features], y[fit])
        if calibrate_intervals:
            self.calibrate(X[~fit], y[~fit])
            self.calibration.update({"split": "random_complete_UTC_days", "seed": CALIBRATION_SEED,
                                     "fraction": CALIBRATION_FRACTION,
                                     "calibration_dates": sorted(str(d.date()) for d in chosen),
                                     "quantile_train_range": [str(X.index[fit].min()), str(X.index[fit].max())],
                                     "n_quantile_train": int(fit.sum())})
        return self

    def calibrate(self, X: pd.DataFrame, y: pd.Series, *, alpha: float = 0.2) -> "PowerModel":
        """CQR на отложенных данных, не использованных для обучения квантилей.

        Группы: лаг × диапазон ветра; для редких/новых групп fallback на лаг, затем общий.
        Scores не используют P50: он обучен и на калибровочном окне.
        """
        if not X.index.equals(y.index):
            raise ValueError("Индексы X и y должны совпадать")
        if not len(y) or not np.isfinite(y).all() or not y.between(0, 1).all():
            raise ValueError("Калибровка требует конечную нормированную мощность в [0, 1]")
        raw = self._predict_raw(X)
        lo = np.minimum(raw["p10"].to_numpy(), raw["p90"].to_numpy())
        hi = np.maximum(raw["p10"].to_numpy(), raw["p90"].to_numpy())
        scores = np.maximum(lo - y.to_numpy(), y.to_numpy() - hi)
        corrections = {"global": {"q": conformal_correction(scores, alpha), "n": len(scores)}}
        leads = X["lead_day"].to_numpy()
        wind_groups = np.digitize(X["wind_speed_100m"], CALIBRATION_WIND_BINS)
        for lead in np.unique(leads):
            for wind in [None, *np.unique(wind_groups[leads == lead])]:
                mask = leads == lead
                key = f"lead{int(lead)}"
                if wind is not None:
                    mask = mask & (wind_groups == wind)
                    key += f"_wind{int(wind)}"
                if mask.sum() >= MIN_CALIBRATION_GROUP:
                    corrections[key] = {"q": conformal_correction(scores[mask], alpha), "n": int(mask.sum())}
        self.calibration = {"method": "split_cqr", "alpha": alpha,
                            "grouping": ["lead_day", "wind_speed_100m_bin"],
                            "wind_bins": list(CALIBRATION_WIND_BINS),
                            "min_group_size": MIN_CALIBRATION_GROUP,
                            "calibration_range": [str(X.index.min()), str(X.index.max())],
                            "n_calibration": len(X), "corrections": corrections}
        return self

    def _predict_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({k: np.clip(m.predict(X[self.features]), 0, 1)
                             for k, m in self.models.items()}, index=X.index)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._predict_raw(X)
        if self.calibration:
            corrections = self.calibration["corrections"]
            shifts = np.array([
                corrections.get(f"lead{int(lead)}_wind{int(wind)}",
                                corrections.get(f"lead{int(lead)}", corrections["global"]))["q"]
                for lead, wind in zip(X["lead_day"], np.digitize(X["wind_speed_100m"], self.calibration["wind_bins"]))
            ])
            lo = np.minimum(out["p10"].to_numpy(), out["p90"].to_numpy())
            hi = np.maximum(out["p10"].to_numpy(), out["p90"].to_numpy())
            out["p10"] = np.clip(lo - shifts, 0, 1)
            out["p90"] = np.clip(hi + shifts, 0, 1)
        # квантили не должны пересекаться
        out["p10"] = np.minimum(out["p10"], out["p50"])
        out["p90"] = np.maximum(out["p90"], out["p50"])
        return out[["p10", "p50", "p90"]]

    def save(self, path: Path = config.ROOT / "docs/research/data/R2_candidate.joblib") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "features": self.features, "meta": self.meta,
                     "calibration": self.calibration}, path)
        return path

    @classmethod
    def load(cls, path: Path = config.ROOT / "docs/research/data/R2_candidate.joblib") -> "PowerModel":
        d = joblib.load(path)
        return cls(models=d["models"], features=d["features"], meta=d.get("meta", {}),
                   calibration=d.get("calibration", {}))
