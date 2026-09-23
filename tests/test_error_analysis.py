"""Regression checks for temporal alignment and misleading R3 aggregations."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("error_analysis", Path(__file__).parents[1] / "scripts/make_error_analysis.py")
r3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3)


def inputs(times, powers, winds=None):
    n = len(times)
    frame = pd.DataFrame({"time": pd.to_datetime(times, utc=True), "turbine": "t1", "lead_day": 1,
                          "power": powers, "p10": 0., "p50": .5, "p90": 1.,
                          "wind_speed_100m": [5.] * n if winds is None else winds, "wind_meas": 5.})
    return frame, frame[r3.KEYS].assign(direction=0.)


def test_ramps_do_not_bridge_gaps_or_daily_forecast_releases():
    frame, direction = inputs(["2025-12-01T18:00Z", "2025-12-01T19:00Z", "2025-12-01T20:00Z",
                               "2025-12-01T22:00Z"], [.4, .7, .3, .9])
    frame["p50"] = [.3, .8, .2, .9]
    b = r3.prepare_predictions(frame.iloc[::-1], direction).reset_index(drop=True)
    assert b.ramp.tolist() == ["unknown", "up", "down", "unknown"]
    assert b.predicted_delta.isna().tolist() == [True, True, False, True]
    assert b.actual_delta.iloc[1] == pytest.approx(.3)
    assert b.predicted_delta.iloc[2] == pytest.approx(-.6)
    assert b.horizon_hour.tolist() == [24, 1, 2, 4]
    assert b.issue_date.iloc[1] == pd.Timestamp("2025-12-01")


def test_regime_boundaries_utc_month_and_both_leads():
    times = pd.date_range("2025-12-31T18:00Z", periods=6, freq="h")
    frame, direction = inputs(times, [0, .05, .3, .7, .95, 1], [0, 3, 6, 9, 12, 45])
    frame["lead_day"] = 2
    direction["lead_day"] = 2
    b = r3.prepare_predictions(frame, direction)
    assert b.wind_regime.astype(str).tolist() == ["<3", "3–6", "6–9", "9–12", "≥12", "≥12"]
    assert b.power_regime.astype(str).tolist() == ["≤0,05", "≤0,05", "0,05–0,3", "0,3–0,7", "0,7–0,95", ">0,95"]
    assert b.month.tolist() == ["2025-12"] + ["2026-01"] * 5
    assert b.horizon_hour.tolist() == [48, 25, 26, 27, 28, 29]
    assert b.covered.all()  # Quantile bounds include exact 0 and 1.


def test_weather_join_rejects_duplicated_or_missing_keys():
    frame, direction = inputs(["2025-12-01T00:00Z", "2025-12-01T01:00Z"], [.1, .2])
    with pytest.raises(pd.errors.MergeError):
        r3.prepare_predictions(frame, pd.concat([direction, direction.iloc[:1]]))
    with pytest.raises(ValueError, match="Missing direction"):
        r3.prepare_predictions(frame, direction.iloc[:1])
    with pytest.raises(ValueError, match="Duplicate forecast key"):
        r3.prepare_predictions(pd.concat([frame, frame.iloc[:1]]), direction)


def test_error_contribution_and_ramp_detection_use_correct_denominators():
    # Three true ramps, two predicted ramps, only one with the right sign.
    frame, direction = inputs(pd.date_range("2025-12-01T00:00Z", periods=4, freq="h"), [.1, .6, .2, .8])
    frame["p50"] = [.1, .6, .7, .2]
    b = r3.prepare_predictions(frame, direction)
    detection = r3.ramp_detection(b).loc[1]
    assert detection.actual_ramps == 3
    assert detection.predicted_ramps == 2
    assert detection.correct_direction_ramps == 1
    assert detection.directional_recall_pct == pytest.approx(100 / 3)
    assert detection.directional_precision_pct == 50
    table = r3.summarize(b, "power_regime")
    assert table.n.sum() == 4
    assert table.error_share_pct.sum() == pytest.approx(100)
    assert np.average(table.mae, weights=table.n) == pytest.approx(b.abs_error.mean())
    assert np.sqrt(np.average(table.rmse**2, weights=table.n)) == pytest.approx(np.sqrt((b.error**2).mean()))
