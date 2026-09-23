"""Обученная модель из models/power_model.joblib: P10/P50/P90 в [0, 1] и без пересечения квантилей."""
from __future__ import annotations

import pytest

from wind_agent import config
from wind_agent.features import make_features
from wind_agent.model import PowerModel

from conftest import model_input

MODEL_PATH = config.MODELS_DIR / "power_model.joblib"


@pytest.fixture(scope="module")
def model() -> PowerModel:
    if not MODEL_PATH.exists():
        pytest.skip("нет models/power_model.joblib — выполните `wind-agent train`")
    return PowerModel.load()


def test_predict_range_and_quantile_order(model, archived_t1):
    pred = model.predict(make_features(model_input(archived_t1, "t1")))
    assert list(pred.columns) == ["p10", "p50", "p90"]
    assert len(pred) == 48
    assert not pred.isna().any().any()
    assert ((pred >= 0) & (pred <= 1)).all().all()
    assert (pred["p10"] <= pred["p50"]).all()
    assert (pred["p50"] <= pred["p90"]).all()


def test_model_meta(model):
    assert set(model.models) == {"p10", "p50", "p90"}
    assert model.meta.get("turbines") == list(config.TURBINES)
