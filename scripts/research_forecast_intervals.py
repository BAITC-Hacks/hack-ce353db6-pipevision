"""Single exploratory lead × predicted-wind CQR candidate; no production changes.

Earlier temporal windows only: train before a two-month calibration window,
then transfer corrections onto the refitted estimator as in production.
This experiment was proposed after inspecting the established final holdout;
its results must not be described as another blind holdout confirmation.
"""
from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd

from wind_agent.model import COVERAGE, PowerModel, build_dataset
from wind_agent.weather import WeatherClient
from research_forecast_improvement import OUT, FOLDS, fit_variant, legacy_dataset

MIN_GROUP = 200


def regimes(X):
    return np.digitize(X.wind_speed_100m.to_numpy(), [3., 9.])


def regime_calibration(model, X, y):
    raw = model.predict_raw(X)
    lo, hi = model._half_widths(raw)
    yy = np.asarray(y, dtype=float)
    scores = np.maximum((raw.p10.to_numpy() - yy) / lo, (yy - raw.p90.to_numpy()) / hi)
    leads, bins = X.lead_day.to_numpy(), regimes(X)
    corrections = {}
    lead_fallback = model.calibrate(X, y)
    for lead in sorted(np.unique(leads)):
        for band in range(3):
            mask = (leads == lead) & (bins == band)
            n = int(mask.sum())
            q = max(0., float(np.quantile(scores[mask], min(1., COVERAGE * (1 + 1 / n))))) if n >= MIN_GROUP else lead_fallback[int(lead)]
            corrections[f'{int(lead)}:{band}'] = {'q': q, 'n': n, 'fallback': n < MIN_GROUP}
    return corrections


def predict_regime(model, X, corrections):
    raw = model.predict_raw(X)
    lo, hi = model._half_widths(raw)
    q = np.array([corrections[f'{int(lead)}:{int(band)}']['q'] for lead, band in zip(X.lead_day, regimes(X))])
    raw['p10'] = np.clip(raw.p10.to_numpy() - q * lo, 0, 1)
    raw['p90'] = np.clip(raw.p90.to_numpy() + q * hi, 0, 1)
    return raw


def score_intervals(X, y, p):
    yy = np.asarray(y, dtype=float)
    width = p.p90.to_numpy() - p.p10.to_numpy()
    score = width + 10 * np.maximum(p.p10.to_numpy() - yy, 0) + 10 * np.maximum(yy - p.p90.to_numpy(), 0)
    covered = (yy >= p.p10.to_numpy()) & (yy <= p.p90.to_numpy())
    groups = {}
    for lead in sorted(X.lead_day.unique()):
        for band in range(3):
            mask = (X.lead_day.to_numpy() == lead) & (regimes(X) == band)
            if mask.any():
                groups[f'{int(lead)}:{band}'] = {'n': int(mask.sum()), 'coverage_pct': float(100 * covered[mask].mean()),
                                              'width': float(width[mask].mean()), 'interval_score': float(score[mask].mean())}
    return {'n': len(X), 'coverage_pct': float(100 * covered.mean()), 'width': float(width.mean()),
            'interval_score': float(score.mean()), 'by': groups,
            'weighted_group_coverage_gap_pp': sum(g['n'] * abs(g['coverage_pct'] - 80) for g in groups.values()) / len(X)}


def main():
    begin = time.monotonic()
    df, X, y = legacy_dataset()
    rows = []
    for cutoff in FOLDS:
        start = pd.Timestamp(cutoff, tz='UTC')
        calstart = start - pd.DateOffset(months=2)
        fit = df.index < calstart
        cal = (df.index >= calstart) & (df.index < start)
        train = df.index < start
        test = (df.index >= start) & (df.index < start + pd.DateOffset(months=2))
        calibration_model = fit_variant(X[fit], y[fit], 'squared_error')
        q = regime_calibration(calibration_model, X[cal], y[cal])
        model = fit_variant(X[train], y[train], 'squared_error')
        model.calibration = calibration_model.calibration
        rows.append({'fold': cutoff, 'calibration_start': str(calstart), 'corrections': q,
                     'baseline': score_intervals(X[test], y[test], model.predict(X[test])),
                     'candidate': score_intervals(X[test], y[test], predict_regime(model, X[test], q))})
        r = rows[-1]
        print(json.dumps({'fold': cutoff, **{k: {m:r[k][m] for m in ['coverage_pct', 'width', 'interval_score', 'weighted_group_coverage_gap_pp']} for k in ['baseline', 'candidate']}}), flush=True)
    ratios = {m: [r['candidate'][m] / r['baseline'][m] for r in rows] for m in ['interval_score', 'width', 'weighted_group_coverage_gap_pp']}
    closer = sum(abs(r['candidate']['coverage_pct']-80) < abs(r['baseline']['coverage_pct']-80) for r in rows)
    accepted = (np.mean(ratios['interval_score']) <= .99 and max(ratios['interval_score']) <= 1.02
                and max(ratios['width']) <= 1.10 and np.mean(ratios['weighted_group_coverage_gap_pp']) < 1 and closer >= 2)
    result = {'exploratory': True, 'protocol': {'wind_thresholds_ms': [3,9], 'min_group': MIN_GROUP,
               'fallback': 'existing lead-only CQR', 'folds': FOLDS, 'calibration_months': 2,
               'gate': 'mean interval score <=99% baseline; no fold score >102%; no fold width >110%; mean relative group calibration gap lower; overall coverage closer to80 in >=2 folds'},
              'rows': rows, 'mean_ratios': {k:float(np.mean(v)) for k,v in ratios.items()},
              'coverage_closer_folds': closer, 'accepted': bool(accepted)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'interval_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print('ACCEPTED', accepted, result['mean_ratios'], 'SECONDS', round(time.monotonic()-begin,2), flush=True)


if __name__ == '__main__':
    main()
