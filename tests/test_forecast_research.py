"""Guard the offline forecast candidates without changing production defaults."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from research_forecast_improvement import MeanMedianBlend, fit_variant, point_metrics
from research_forecast_intervals import regime_calibration, predict_regime, score_intervals
from wind_agent.model import PowerModel


class ConstantRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def frame(n=800):
    X = pd.DataFrame({'wind_speed_100m': np.tile(np.repeat([2., 6.], n // 4), 2),
                      'lead_day': np.repeat([1, 2], n // 2)},
                     index=pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC'))
    return X


def model():
    return PowerModel(models={k: ConstantRegressor(v) for k, v in [('p10', .2), ('p50', .5), ('p90', .8)]},
                      features=['wind_speed_100m', 'lead_day'])


def test_blend_clips_members_before_weighting_and_roundtrips(tmp_path):
    X = frame()
    blend = MeanMedianBlend(ConstantRegressor(1.2), ConstantRegressor(-.3), .25)
    m = model()
    m.models['p50'] = blend
    m.calibration = {1: .2, 2: .3}
    expected = m.predict(X)
    assert np.allclose(expected.p50, .75)
    loaded = PowerModel.load(m.save(tmp_path / 'candidate.joblib'))
    assert loaded.calibration == m.calibration
    pd.testing.assert_frame_equal(expected, loaded.predict(X))
    with pytest.raises(ValueError):
        MeanMedianBlend(ConstantRegressor(.5), ConstantRegressor(.5), 1.5)
    with pytest.raises(ValueError):
        fit_variant(X, pd.Series(.5, index=X.index), 'typo')


def test_regime_calibration_uses_only_given_labels_and_falls_back_for_sparse_groups():
    X = frame()
    y = pd.Series(np.where(X.wind_speed_100m < 3, .1, .95), index=X.index)
    m = model()
    q = regime_calibration(m, X, y)
    assert q['1:0']['q'] == pytest.approx(1 / 3)
    assert q['1:1']['q'] == pytest.approx(.5)
    assert q['1:2']['fallback'] is True
    assert q['1:2']['q'] == m.calibration[1]
    assert q['1:2']['n'] == 0
    forecast = predict_regime(m, X, q)
    assert np.allclose(forecast.p50, .5)
    assert (forecast.p10 <= forecast.p50).all()
    assert (forecast.p50 <= forecast.p90).all()
    assert forecast.p10.min() >= 0 and forecast.p90.max() <= 1


def test_interval_score_penalizes_width_and_missing_tails():
    X = frame()
    y = np.full(len(X), .5)
    narrow = pd.DataFrame({'p10': .4, 'p50': .5, 'p90': .6}, index=X.index)
    wide = narrow.assign(p10=0., p90=1.)
    missed = narrow.assign(p10=.7, p90=.9)
    assert score_intervals(X, y, narrow)['interval_score'] == pytest.approx(.2)
    assert score_intervals(X, y, wide)['interval_score'] == pytest.approx(1.)
    assert score_intervals(X, y, missed)['interval_score'] == pytest.approx(2.2)


def test_ramp_comparison_does_not_bridge_missing_hour_or_new_issue():
    index = pd.to_datetime(['2025-01-01T17:00Z', '2025-01-01T18:00Z', '2025-01-01T19:00Z', '2025-01-01T21:00Z'])
    df = pd.DataFrame({'turbine': 't1', 'lead_day': 1, 'power': [.1, .5, .1, .9]}, index=index)
    df.index.name = 'time'
    result = point_metrics(df, np.array([.1, .5, .1, .9]))
    # Local midnight is UTC19:00; its prediction is a different issue, and 21:00 has a gap.
    assert result['ramps']['up']['n'] == 1
    assert result['ramps']['down']['n'] == 1
    assert result['ramps']['unknown']['n'] == 2
    assert result['ramp_detection'][0]['eligible_transitions'] == 1
    assert result['ramp_detection'][0]['correct_direction_ramps'] == 1
