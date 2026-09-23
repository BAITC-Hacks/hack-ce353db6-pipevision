"""Offline checks for CQR, leakage boundaries and compatibility with saved models."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import joblib
import numpy as np
import pandas as pd
import pytest

from r2_candidate import PowerModel, conformal_correction


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def fixture_model():
    return PowerModel(models={k: ConstantModel(v) for k, v in
                              [('p10', .2), ('p50', .5), ('p90', .8)]},
                      features=['lead_day', 'turbine_id', 'wind_speed_100m'])


def frame(n=200):
    return pd.DataFrame({'lead_day': np.repeat([1, 2], n // 2), 'turbine_id': 0, 'wind_speed_100m': 2.0},
                        index=pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC'))


def test_finite_sample_rank_and_small_sample():
    assert conformal_correction(np.arange(9) / 10) == pytest.approx(.7)
    assert conformal_correction(np.array([.1, .2])) == 1
    assert conformal_correction(np.full(200, -.1)) == pytest.approx(-.1)
    for invalid in [[], [np.nan], [np.inf]]:
        with pytest.raises(ValueError):
            conformal_correction(invalid)


def test_lag_corrections_p50_and_fallback():
    X = frame()
    model = fixture_model()
    before = model.predict(X)
    y = pd.Series(np.repeat([.05, .98], 100), index=X.index)
    model.calibrate(X, y)
    after = model.predict(X)
    pd.testing.assert_series_equal(after.p50, before.p50)
    assert after.p10.iloc[0] == pytest.approx(.05)
    assert after.p90.iloc[-1] == pytest.approx(.98)
    assert model.calibration['corrections']['lead1_wind0']['q'] == pytest.approx(.15)
    unknown = X.iloc[:2].copy()
    unknown['wind_speed_100m'] = 19
    assert model.predict(unknown).p10.iloc[0] == pytest.approx(.05)
    unknown['lead_day'] = 9
    pred = model.predict(unknown)
    assert pred.p10.iloc[0] == pytest.approx(.02)  # global fallback


def test_duplicate_indices_and_shrinking_do_not_move_p50():
    X = frame()
    X.index = pd.DatetimeIndex(np.repeat(X.index[:100], 2))
    model = fixture_model()
    y = pd.Series(.5, index=X.index)
    model.calibrate(X, y)
    pred = model.predict(X)
    assert pred.index.equals(X.index)
    np.testing.assert_allclose(pred, .5)
    assert list(pred.columns) == ['p10', 'p50', 'p90']


def test_save_load_and_old_artifact(tmp_path):
    X = frame()
    model = fixture_model().calibrate(X, pd.Series(.99, index=X.index))
    path = model.save(tmp_path / 'calibrated.joblib')
    restored = PowerModel.load(path)
    assert restored.calibration == model.calibration
    pd.testing.assert_frame_equal(restored.predict(X), model.predict(X))
    old = tmp_path / 'old.joblib'
    joblib.dump({'models': model.models, 'features': model.features}, old)
    legacy = PowerModel.load(old)
    assert legacy.calibration == {}
    assert legacy.predict(X).p10.iloc[0] == .2


def test_temporal_split_keeps_all_rows_of_hour_together_and_refit_clears(monkeypatch):
    calls = []

    class RecordingRegressor(ConstantModel):
        def __init__(self, loss, quantile=.5, **kwargs):
            super().__init__(quantile)
            self.loss = loss

        def fit(self, X, y):
            calls.append((self.loss, X.index.copy()))
            return self

    monkeypatch.setattr('r2_candidate.HistGradientBoostingRegressor', RecordingRegressor)
    X = frame(960)
    X.index = pd.DatetimeIndex(np.repeat(pd.date_range('2025-01-01', periods=480, freq='h', tz='UTC'), 2))
    y = pd.Series(.5, index=X.index)
    model = fixture_model().fit(X, y, calibrate_intervals=True)
    assert calls[0][1].equals(X.index)
    for _, index in calls[1:]:
        cal_dates = set(model.calibration['calibration_dates'])
        assert not set(index.strftime('%Y-%m-%d')) & cal_dates
        assert len(index) == 768
    assert model.calibration['n_calibration'] == 192
    model.fit(X, y)
    assert model.calibration == {}


def test_invalid_calibration_inputs():
    X = frame()
    model = fixture_model()
    with pytest.raises(ValueError):
        model.calibrate(X, pd.Series(1.1, index=X.index))
    with pytest.raises(ValueError):
        model.calibrate(X, pd.Series(.5, index=X.index[::-1]))
    with pytest.raises(ValueError):
        model.fit(X.iloc[:10], pd.Series(.5, index=X.index[:10]), calibrate_intervals=True)


def test_wind_regimes_use_different_corrections():
    X = frame(400)
    X['lead_day'] = 1
    X['wind_speed_100m'] = np.repeat([2.0, 10.0], 200)
    y = pd.Series(np.repeat([.1, .99], 200), index=X.index)
    model = fixture_model().calibrate(X, y)
    q = model.calibration['corrections']
    assert q['lead1_wind0']['q'] == pytest.approx(.1)
    assert q['lead1_wind2']['q'] == pytest.approx(.19)
    p = model.predict(X)
    assert p.p10.iloc[0] == pytest.approx(.1)
    assert p.p10.iloc[-1] == pytest.approx(.01)
