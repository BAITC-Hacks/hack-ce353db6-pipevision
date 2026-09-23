"""Матрица признаков из архивного прогноза погоды."""
from __future__ import annotations

import pandas as pd
import pytest

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


@pytest.mark.parametrize("time_unit", ["us", "ns"])
def test_weather_neighbors_do_not_jump_gaps_or_depend_on_row_order(time_unit):
    from wind_agent.data import weather_neighbors
    t = pd.to_datetime(['2025-01-01T00:00Z', '2025-01-01T01:00Z', '2025-01-01T03:00Z']).as_unit(time_unit)
    df = pd.DataFrame({'wind_speed_100m': [2., 5., 9.], 'lead_day': 1, 'turbine': 't1'}, index=t)
    result = weather_neighbors(df)
    assert result.ws100_prev.tolist() == [2., 2., 9.]
    assert result.ws100_next.tolist() == [5., 5., 9.]
    pd.testing.assert_frame_equal(weather_neighbors(df.iloc[::-1]).iloc[::-1], result)


def test_weather_neighbors_respect_release_and_lead_boundaries():
    from wind_agent.data import weather_neighbors
    t = pd.to_datetime(['2025-01-01T18:00Z', '2025-01-01T19:00Z'])
    df = pd.DataFrame({'wind_speed_100m': [2., 9.], 'lead_day': 1, 'turbine': 't1'}, index=t)
    # Nominal 23:00 issue: these target hours belong to different local days/releases.
    assert weather_neighbors(df).ws100_next.tolist() == [2., 9.]
    # Custom noon issue spans local midnight: one release may use the exact neighbor.
    df['issue_time_utc'] = pd.Timestamp('2025-01-01T07:00Z')
    assert weather_neighbors(df).ws100_next.tolist() == [9., 9.]
    df['lead_day'] = [1, 2]
    assert weather_neighbors(df).ws100_next.tolist() == [2., 9.]


@pytest.mark.parametrize("rejection", ["missing", "few_samples", "curtailment"])
def test_training_neighbor_uses_weather_when_scada_hour_is_rejected(monkeypatch, rejection):
    from wind_agent import config, data
    t = pd.date_range('2025-01-01', periods=3, freq='h', tz='UTC')
    scada = pd.DataFrame({'power': [.1,.2,.3], 'wind_meas': 5., 'temp_meas': 10.,
                          'n': [6, 6, 6], 'curtailed_frac': 0.}, index=t)
    if rejection == 'missing':
        scada = scada.drop(t[1])
    elif rejection == 'few_samples':
        scada.loc[t[1], 'n'] = 1
    else:
        scada.loc[t[1], 'curtailed_frac'] = 1.
    monkeypatch.setattr(data, 'load_hourly', lambda _: scada)

    class Weather:
        def previous_runs(self, *args, **kwargs):
            result = pd.DataFrame({'time': t})
            for lead in config.LEAD_DAYS:
                for v in config.WEATHER_VARS:
                    result[f'{v}_previous_day{lead}'] = [2., 5., 9.] if v.startswith('wind_speed') else 10.
            return result

        def ensemble_previous_runs(self, *args, **kwargs):
            return {}

    df = data.training_frame('t1', Weather())
    assert len(df) == 4  # middle actual rejected for both leads
    X = make_features(df)
    assert X.ws100_next[df.index == t[0]].tolist() == [5., 5.]
    assert X.ws100_prev[df.index == t[2]].tolist() == [5., 5.]
    # The full forecast input at inference has the same context for surviving targets.
    full = Weather().previous_runs().set_index('time')
    for lead in config.LEAD_DAYS:
        f = full.rename(columns={f'{v}_previous_day{lead}': v for v in config.WEATHER_VARS})
        f = f.assign(lead_day=lead, turbine='t1', issue_time_utc=pd.Timestamp('2024-12-31T18:00Z'))
        expected = make_features(f).loc[[t[0],t[2]], ['ws100_prev','ws100_next']]
        actual = X.loc[df.lead_day == lead, ['ws100_prev','ws100_next']]
        pd.testing.assert_frame_equal(actual, expected, check_names=False)


def test_weather_context_skips_repeated_local_hour_at_kazakhstan_clock_change():
    from wind_agent.data import weather_neighbors
    # UTC18 is a second local23:00: present in NWP, absent from the serving local grid.
    t = pd.to_datetime(['2024-02-29T17:00Z', '2024-02-29T18:00Z', '2024-02-29T19:00Z'])
    full = pd.DataFrame({'wind_speed_100m': [2., 5., 9.], 'lead_day': 1, 'turbine': 't1'}, index=t)
    training_context = weather_neighbors(full).loc[[t[0], t[2]]]
    serving = full.loc[[t[0], t[2]]].assign(lead_day=[1, 2], issue_time_utc=pd.Timestamp('2024-02-28T17:00Z'))
    pd.testing.assert_frame_equal(training_context, weather_neighbors(serving))
    assert training_context.ws100_next.tolist() == [2., 9.]
