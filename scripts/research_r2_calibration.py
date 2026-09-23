"""Reproduce R2 with cached weather: PYTHONPATH=src python scripts/research_r2_calibration.py.

Default: baseline vs the fixed CQR configuration, JSON and PNG report artifacts.
--validation: also rerun the 16 CQR candidates and the hierarchical variant on
three pre-test validation periods.
Corrections use only pre-test data. The target backtest was inspected twice during
research; it is not an untouched lockbox (see the report).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from wind_agent import config
from wind_agent.model import HGB_PARAMS, build_dataset, metrics
from r2_candidate import PowerModel, conformal_correction
from wind_agent.weather import WeatherClient

OUT = config.ROOT / 'docs' / 'research'


def summarize(df, pred):
    frame = df[['power', 'turbine', 'lead_day']].copy()
    frame[['p10', 'p50', 'p90']] = pred

    def one(g):
        return {**metrics(g.power, g.p50),
                'coverage_pct': float(100 * ((g.power >= g.p10) & (g.power <= g.p90)).mean()),
                'width': float((g.p90 - g.p10).mean())}

    return {'overall': one(frame),
            'by': {f'{t}_lead{lead}': one(g) for (t, lead), g in frame.groupby(['turbine', 'lead_day'])}}


def compare(df, X, y):
    tr = df.index < pd.Timestamp(config.BACKTEST_TEST_START, tz='UTC')
    te = (df.index >= pd.Timestamp(config.BACKTEST_TEST_START, tz='UTC')) & (
        df.index < pd.Timestamp(config.BACKTEST_TEST_END, tz='UTC') + pd.Timedelta(days=1))
    baseline = PowerModel().fit(X[tr], y[tr]).predict(X[te])
    model = PowerModel().fit(X[tr], y[tr], calibrate_intervals=True)
    pred = model.predict(X[te])
    assert baseline.p50.equals(pred.p50), 'CQR must leave the point forecast exactly unchanged'
    out = {'baseline': summarize(df[te], baseline), 'calibrated': summarize(df[te], pred),
           'p50_identical': True, 'calibration': model.calibration}
    (OUT / 'data' / 'R2_comparison.json').write_text(json.dumps(out, indent=2) + '\n')
    labels = list(out['baseline']['by'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(labels))
    for offset, name, title in [(-.18, 'baseline', 'Before'), (.18, 'calibrated', 'CQR')]:
        rows = [out[name]['by'][key] for key in labels]
        for ax, field in zip(axes, ['coverage_pct', 'width']):
            values = [r[field] for r in rows]
            bars = ax.bar(x + offset, values, .36, label=title)
            ax.bar_label(bars, fmt='%.1f' if field == 'coverage_pct' else '%.3f', fontsize=8, padding=3)
    axes[0].axhspan(78, 82, color='green', alpha=.12, label='Target 78–82%')
    axes[0].axhline(80, color='green', linestyle='--', linewidth=1)
    axes[0].set(ylabel='P10–P90 coverage, %', ylim=(0, 100))
    axes[1].set(ylabel='Mean interval width, rated power', ylim=(0, .65))
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.legend(fontsize=8, ncol=3, loc='upper center')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('R2 · December 2025–January 2026 · P50 unchanged')
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / 'R2_coverage.png', dpi=160)
    plt.close(fig)
    print(json.dumps({k: out[k]['overall'] for k in ['baseline', 'calibrated']}, indent=2), flush=True)


def validate(df, X, y):
    results = []
    for end in ['2024-11-30', '2025-07-31', '2025-09-30']:
        cutoff = pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=1)
        train = df.index < cutoff
        val = (df.index >= cutoff) & (df.index < cutoff + pd.DateOffset(months=2))
        mid = HistGradientBoostingRegressor(loss='squared_error', **HGB_PARAMS).fit(X[train], y[train]).predict(X[val]).clip(0, 1)
        for days in [30, 60, 90, 0]:
            if days:
                cal = (df.index >= cutoff - pd.Timedelta(days=days)) & train
            else:
                dates = df.index.normalize()
                unique = dates[train].unique().sort_values()
                chosen = np.random.default_rng(42).choice(unique, size=math.ceil(len(unique) * .2), replace=False)
                cal = dates.isin(chosen) & train
            fit = train & ~cal
            bounds = {}
            for name, q in [('lo', .1), ('hi', .9)]:
                model = HistGradientBoostingRegressor(loss='quantile', quantile=q, **HGB_PARAMS).fit(X[fit], y[fit])
                bounds[name] = [model.predict(X[cal]).clip(0, 1), model.predict(X[val]).clip(0, 1)]
            clo, chi = np.minimum(bounds['lo'][0], bounds['hi'][0]), np.maximum(bounds['lo'][0], bounds['hi'][0])
            vlo, vhi = np.minimum(bounds['lo'][1], bounds['hi'][1]), np.maximum(bounds['lo'][1], bounds['hi'][1])
            scores = np.maximum(clo - y[cal].to_numpy(), y[cal].to_numpy() - chi)
            groupings = ['lead', 'lead_turbine', 'lead_wind', 'lead_turbine_wind']
            if not days:
                groupings.append('lead_wind_hierarchical')
            for grouping in groupings:
                def keys(mask):
                    f = X[mask]
                    key = f.lead_day.astype(str)
                    if 'turbine' in grouping:
                        key = key + '_' + f.turbine_id.astype(str)
                    if 'wind' in grouping:
                        key = key + '_' + pd.Series(np.digitize(f.wind_speed_100m, [4, 8, 12]), index=f.index).astype(str)
                    return key.to_numpy()
                ck, vk = keys(cal), keys(val)
                shifts = {}
                for k in np.unique(ck):
                    s = scores[ck == k]
                    if 'hierarchical' in grouping:
                        # HCP: equal total weight per UTC day; one extra day's mass at infinity.
                        dates = df.index[cal][ck == k].normalize()
                        _, inverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
                        weights = 1.0 / counts[inverse]
                        order = np.argsort(s)
                        pos = np.searchsorted(np.cumsum(weights[order]), .8 * (len(counts) + 1), side='left')
                        shifts[k] = float(s[order[pos]]) if pos < len(s) else 1.0
                    else:
                        shifts[k] = conformal_correction(s)
                delta = np.array([shifts[k] for k in vk])
                lo, hi = np.minimum((vlo - delta).clip(0, 1), mid), np.maximum((vhi + delta).clip(0, 1), mid)
                actual = y[val].to_numpy()
                covered = (actual >= lo) & (actual <= hi)
                groups = []
                for t in [0, 1]:
                    for lead in [1, 2]:
                        mask = (X[val].turbine_id == t).to_numpy() & (X[val].lead_day == lead).to_numpy()
                        groups.append(round(float(100 * covered[mask].mean()), 3))
                row = dict(end=end, days=days, grouping=grouping, coverage=float(100 * covered.mean()),
                           width=float((hi - lo).mean()), by=groups)
                results.append(row)
                print(json.dumps(row), flush=True)
    ordinary = [r for r in results if 'hierarchical' not in r['grouping']]
    hierarchical = [{**r, 'grouping': 'lead_wind'} for r in results if 'hierarchical' in r['grouping']]
    (OUT / 'data' / 'R2_validation.json').write_text(json.dumps(ordinary, indent=2) + '\n')
    (OUT / 'data' / 'R2_hierarchical_validation.json').write_text(json.dumps(hierarchical, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    (OUT / 'data').mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    df, X, y = build_dataset(WeatherClient(offline=True))
    if args.validation:
        validate(df, X, y)
    compare(df, X, y)
