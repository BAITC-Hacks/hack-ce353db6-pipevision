"""Offline, time-ordered comparison of mean/median power forecasts.

Selection uses only three windows ending before the established final holdout.
Run with OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src.
No production artifact or existing research report is overwritten.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from wind_agent import config
from wind_agent.model import HGB_PARAMS, PowerModel, build_dataset, metrics
from wind_agent.weather import WeatherClient
from wind_agent.data import utc_to_local
from make_error_analysis import add_ramps, ramp_detection

def legacy_dataset():
    """Freeze pre-fix neighbor semantics so historical comparisons remain reproducible."""
    df, X, y = build_dataset(WeatherClient(offline=True))
    ws = df['wind_speed_100m'].astype(float)
    groups = [df['turbine'].to_numpy(), df['lead_day'].to_numpy()]
    X['ws100_prev'] = ws.groupby(groups).shift(1).fillna(ws)
    X['ws100_next'] = ws.groupby(groups).shift(-1).fillna(ws)
    return df, X, y


OUT = config.ROOT / 'docs/research/data/forecast_improvement'
FOLDS = ['2024-12-01', '2025-08-01', '2025-10-01']
VARIANTS = ['squared_error', 'absolute_error', 'mean_median_blend', 'median_quarter', 'squared_leaf15']


def point_models(X, y, features):
    models = {loss: HistGradientBoostingRegressor(loss=loss, **HGB_PARAMS).fit(X[features], y)
              for loss in VARIANTS[:2]}
    models['squared_leaf15'] = HistGradientBoostingRegressor(loss='squared_error', **{**HGB_PARAMS, 'max_leaf_nodes': 15}).fit(X[features], y)
    return models


def point_predictions(models, X, features):
    result = {k: m.predict(X[features]).clip(0, 1) for k, m in models.items()}
    result['mean_median_blend'] = .5 * (result['squared_error'] + result['absolute_error'])
    result['median_quarter'] = .75 * result['squared_error'] + .25 * result['absolute_error']
    return result


def point_metrics(df, pred):
    result = metrics(pd.Series(df.power.to_numpy()), pd.Series(pred))
    b = df[['turbine', 'lead_day', 'power']].reset_index().rename(columns={'index': 'time'})
    b['p50'] = pred
    local = utc_to_local(pd.DatetimeIndex(b.time))
    b['issue_date'] = local.normalize() - pd.to_timedelta(b.lead_day.to_numpy(), unit='D')
    b = add_ramps(b)
    result['ramps'] = {k: metrics(g.power, g.p50) for k, g in b.groupby('ramp')}
    result['ramp_mae'] = float((b.loc[b.ramp.isin(['up', 'down']), 'p50'] - b.loc[b.ramp.isin(['up', 'down']), 'power']).abs().mean())
    result['by'] = {f'{t}_lead{lead}': metrics(g.power, g.p50)
                    for (t, lead), g in b.groupby(['turbine', 'lead_day'])}
    result['ramp_detection'] = ramp_detection(b).reset_index().to_dict(orient='records')
    return result


def validation(df, X, y):
    rows = []
    for cutoff in FOLDS:
        start = pd.Timestamp(cutoff, tz='UTC')
        tr = df.index < start
        te = (df.index >= start) & (df.index < start + pd.DateOffset(months=2))
        models = point_models(X[tr], y[tr], list(X.columns))
        predictions = point_predictions(models, X[te], list(X.columns))
        for name, p in predictions.items():
            row = {'fold': cutoff, 'variant': name, **point_metrics(df[te], p)}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ['fold', 'variant', 'mae', 'rmse', 'bias', 'ramp_mae']}), flush=True)
    baselines = {r['fold']: r for r in rows if r['variant'] == 'squared_error'}
    summary = {}
    for name in VARIANTS:
        records = [r for r in rows if r['variant'] == name]
        ratios = {k: [r[k] / baselines[r['fold']][k] for r in records]
                  for k in ['mae', 'rmse', 'ramp_mae']}
        eligible = (max(ratios['mae']) <= 1.01 and np.mean(ratios['rmse']) <= 1.02
                    and np.mean(ratios['ramp_mae']) <= 1.02)
        summary[name] = {'mean_relative_mae': float(np.mean(ratios['mae'])),
                         'mean_relative_rmse': float(np.mean(ratios['rmse'])),
                         'mean_relative_ramp_mae': float(np.mean(ratios['ramp_mae'])),
                         'eligible': bool(eligible)}
    selected = min((k for k, v in summary.items() if v['eligible']),
                   key=lambda k: summary[k]['mean_relative_mae'])
    result = {'protocol': {'folds': FOLDS, 'months_per_fold': 2, 'variants': VARIANTS,
                'selection': 'min mean relative MAE; each fold MAE <= baseline*1.01; mean relative RMSE and ramp MAE <=1.02',
                'holdout_promotion': 'MAE lower, RMSE and ramp MAE <=baseline*1.02, no turbine-lead MAE worse, coverage not >1pp lower',
                'features': list(X.columns), 'params': HGB_PARAMS,
                'exploration': 'First mean/median/50-50 blend screened; median and blend failed ramp constraint. Added quarter-median and squared leaf15 before opening holdout; same frozen gate.',
                'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__}},
              'rows': rows, 'summary': summary, 'selected': selected}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('SELECTED', selected, summary, flush=True)
    return selected


class MeanMedianBlend:
    """Research-only estimator; production implementation is independent."""
    def __init__(self, mean_model, median_model, weight=.5):
        if not 0 <= weight <= 1:
            raise ValueError("median weight must be in [0, 1]")
        self.mean_model, self.median_model, self.weight = mean_model, median_model, weight

    def predict(self, X):
        return (1 - self.weight) * np.clip(self.mean_model.predict(X), 0, 1) + self.weight * np.clip(self.median_model.predict(X), 0, 1)


def fit_variant(X, y, variant, baseline=None):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown point variant: {variant}')
    if baseline is None:
        # Pin the old mean estimator independently of production PowerModel.fit.
        estimators = {'p50': HistGradientBoostingRegressor(loss='squared_error', **HGB_PARAMS).fit(X, y)}
        for name, q in [('p10', .1), ('p90', .9)]:
            estimators[name] = HistGradientBoostingRegressor(loss='quantile', quantile=q, **HGB_PARAMS).fit(X, y)
        baseline = PowerModel(models=estimators, features=list(X.columns))
    model = baseline
    if variant == 'squared_error':
        return model
    if variant == 'squared_leaf15':
        point = HistGradientBoostingRegressor(loss='squared_error', **{**HGB_PARAMS, 'max_leaf_nodes': 15}).fit(X[model.features], y)
        return PowerModel(models={**model.models, 'p50': point}, features=list(model.features))
    median = HistGradientBoostingRegressor(loss='absolute_error', **HGB_PARAMS).fit(X[model.features], y)
    # Keep quantile estimators unchanged; only P50 and its ordering-dependent CQR widths differ.
    return PowerModel(models={**model.models, 'p50': median if variant == 'absolute_error' else MeanMedianBlend(model.models['p50'], median, .25 if variant == 'median_quarter' else .5)},
                      features=list(model.features))


def heldout(df, X, y, selected):
    tr = df.index < pd.Timestamp(config.BACKTEST_TEST_START, tz='UTC')
    te = (df.index >= pd.Timestamp(config.BACKTEST_TEST_START, tz='UTC')) & (df.index < pd.Timestamp(config.BACKTEST_TEST_END, tz='UTC') + pd.Timedelta(days=1))
    cal = tr & (df.index >= pd.Timestamp(config.BACKTEST_CALIB_START, tz='UTC'))
    fit_only = tr & ~cal
    baseline_cal = fit_variant(X[fit_only], y[fit_only], 'squared_error')
    baseline = fit_variant(X[tr], y[tr], 'squared_error')
    records = {}
    for variant in dict.fromkeys(['squared_error', selected]):
        c = baseline_cal if variant == 'squared_error' else fit_variant(X[fit_only], y[fit_only], variant, baseline_cal)
        m = baseline if variant == 'squared_error' else fit_variant(X[tr], y[tr], variant, baseline)
        m.calibration = c.calibrate(X[cal], y[cal])
        p = m.predict(X[te])
        g = df[te].copy()
        g[['p10', 'p50', 'p90']] = p
        row = point_metrics(g, p.p50.to_numpy())
        row['calibration'] = m.calibration
        row['coverage_pct'] = float(100 * g.power.between(g.p10, g.p90).mean())
        row['width'] = float((g.p90 - g.p10).mean())
        row['sector'] = {str(s): metrics(h.power, h.p50) for s, h in g.groupby((np.floor(((g.wind_direction_100m % 360) + 11.25) / 22.5).astype(int) % 16))}
        row['wind'] = {str(s): metrics(h.power, h.p50) for s, h in g.groupby(pd.cut(g.wind_speed_100m, [0,3,6,9,12,np.inf], right=False), observed=True)}
        row['power'] = {str(s): metrics(h.power, h.p50) for s, h in g.groupby(pd.cut(g.power, [-np.inf,.05,.3,.7,.95,np.inf]), observed=True)}
        records[variant] = row
        g[['turbine', 'lead_day', 'power', 'p10', 'p50', 'p90', 'wind_speed_100m', 'wind_direction_100m']].to_csv(OUT / f'holdout_{variant}.csv')
        print('HOLDOUT', variant, json.dumps(row), flush=True)
    b, c = records['squared_error'], records[selected]
    accepted = (selected != 'squared_error' and c['mae'] < b['mae'] and c['rmse'] <= b['rmse'] * 1.02
                and c['ramp_mae'] <= b['ramp_mae'] * 1.02
                and all(c['by'][k]['mae'] <= b['by'][k]['mae'] for k in b['by'])
                and c['coverage_pct'] >= b['coverage_pct'] - 1)
    result = {'selected': selected, 'passes_numeric_gate': bool(accepted), 'production_promoted': False, 'records': records}
    (OUT / 'holdout.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASSES_NUMERIC_GATE', accepted, 'PRODUCTION_PROMOTED', False, flush=True)
    return result


def paired_uncertainty(selected):
    """Paired circular seven-day block bootstrap; keep both leads/turbines together."""
    base = pd.read_csv(OUT / 'holdout_squared_error.csv')
    candidate = pd.read_csv(OUT / f'holdout_{selected}.csv')
    keys = ['time', 'turbine', 'lead_day']
    if not base[keys].equals(candidate[keys]):
        raise ValueError('Holdout prediction keys differ')
    days = (pd.to_datetime(base.time, utc=True) + pd.Timedelta(hours=5)).dt.strftime('%Y-%m-%d')
    daily = pd.DataFrame({'day': days,
        'sum_difference': (candidate.p50 - candidate.power).abs() - (base.p50 - base.power).abs(),
        'n': 1}).groupby('day').sum()
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(2000):
        starts = rng.integers(len(daily), size=(len(daily) + 6) // 7)
        positions = np.concatenate([(s + np.arange(7)) % len(daily) for s in starts])[:len(daily)]
        sample = daily.iloc[positions]
        differences.append(sample.sum_difference.sum() / sample.n.sum())
    result = {'block_days': 7, 'resamples': 2000, 'random_seed': 42, 'day_count': len(daily),
              'mae_candidate_minus_baseline_95pct': np.quantile(differences, [.025, .975]).tolist(),
              'interpretation': 'Conditional exploratory uncertainty on this 62-day sample; does not establish generalization to new seasons or sites.'}
    (OUT / 'paired_uncertainty.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout', action='store_true', help='Run the previously frozen selected variant once against final test')
    args = parser.parse_args()
    begin = time.monotonic()
    df, X, y = legacy_dataset()
    print('DATASET', len(df), str(df.index.min()), str(df.index.max()), flush=True)
    if args.holdout:
        selected = json.loads((OUT / 'validation.json').read_text())['selected']
        heldout(df, X, y, selected)
        paired_uncertainty(selected)
    else:
        validation(df, X, y)
    print('SECONDS', round(time.monotonic() - begin, 2), flush=True)


if __name__ == '__main__':
    main()
