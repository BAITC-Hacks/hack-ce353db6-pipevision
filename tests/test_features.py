"""Матрица признаков из архивного прогноза погоды."""
from __future__ import annotations

import pandas as pd

from wind_agent.features import FEATURES, make_features

from conftest import ISSUE_DATE, model_input


def test_make_features_columns_and_no_nan(archived_t1):
    X = make_features(model_input(archived_t1, "t1"))
    assert list(X.columns) == FEATURES
    assert len(X) == 48
    assert not X.isna().any().any()
    assert set(X["lead_day"]) == {1, 2}
    assert (X["turbine_id"] == 0).all()


def test_make_features_two_turbines(wx, archived_t1):
    fc2 = wx.archived_forecast("t2", ISSUE_DATE)
    df = pd.concat([model_input(archived_t1, "t1"), model_input(fc2, "t2")])
    X = make_features(df)
    assert list(X.columns) == FEATURES
    assert len(X) == 96
    assert not X.isna().any().any()
    assert set(X["turbine_id"]) == {0, 1}
