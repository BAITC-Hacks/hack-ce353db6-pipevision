"""Offline R1 ablations, directional statistics and figures.

PYTHONPATH=src python scripts/research_r1_terrain.py [--backtests] [--validation]
The production calibration protocol is reused unchanged for every backtest.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from wind_agent import config, model
from wind_agent.features import FEATURES, MODEL_TERRAIN_FEATURES, make_features
from wind_agent.terrain import (TERRAIN_FEATURES, directional_features, load_context,
                                to_local, polygons, sector_index, wake_counts)
from wind_agent.weather import WeatherClient
from wind_agent.data import utc_to_local

OUT = config.ROOT / 'docs/research'
DATA = OUT / 'data'
FIGURES = OUT / 'figures'
BASE_FEATURES = [f for f in FEATURES if f not in TERRAIN_FEATURES]
SECTORS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
VARIANTS = {'baseline': BASE_FEATURES, 'all': BASE_FEATURES + TERRAIN_FEATURES,
            'selected': BASE_FEATURES + MODEL_TERRAIN_FEATURES}


def all_features(df):
    X = make_features(df)
    terrain = directional_features(df)
    for col in TERRAIN_FEATURES:
        X[col] = terrain[col].to_numpy()
    return X[BASE_FEATURES + TERRAIN_FEATURES]


def backtests():
    # PowerModel's dataclass factory reads model.FEATURES at instantiation.
    # Only this research process changes those globals; restore even on failure.
    old_features, old_make = model.FEATURES, model.make_features
    try:
        model.make_features = all_features
        for name, cols in VARIANTS.items():
            model.FEATURES = cols
            with tempfile.TemporaryDirectory(prefix='wind-r1-') as tmp:
                result = model.backtest(WeatherClient(offline=True), out_dir=Path(tmp))
                for suffix in ['metrics.json', 'predictions.csv']:
                    shutil.copyfile(Path(tmp) / f'backtest_{suffix}', DATA / f'R1_{name}_{suffix}')
                print(name, result['overall']['model'], flush=True)
    finally:
        model.FEATURES, model.make_features = old_features, old_make


def validation(df, X, y):
    blocks = {'wake': TERRAIN_FEATURES[:2], 'land': TERRAIN_FEATURES[2:4], 'slope': TERRAIN_FEATURES[4:]}
    rows = []
    for cutoff in ['2024-12-01', '2025-08-01', '2025-10-01']:
        start = pd.Timestamp(cutoff, tz='UTC')
        tr = df.index < start
        val = (df.index >= start) & (df.index < start + pd.DateOffset(months=2))
        for n in range(4):
            for combo in itertools.combinations(blocks, n):
                cols = BASE_FEATURES + sum([blocks[b] for b in combo], [])
                estimator = HistGradientBoostingRegressor(loss='squared_error', **model.HGB_PARAMS).fit(X.loc[tr, cols], y[tr])
                pred = estimator.predict(X.loc[val, cols]).clip(0, 1)
                error = pred - y[val].to_numpy()
                row = {'cutoff': cutoff, 'variant': '+'.join(combo) or 'baseline',
                       'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt((error**2).mean())), 'by': {}}
                for (t, lead), g in df[val].reset_index().groupby(['turbine', 'lead_day']):
                    row['by'][f'{t}_lead{lead}'] = float(np.abs(error[g.index]).mean())
                rows.append(row)
                print(json.dumps(row), flush=True)
    (DATA / 'R1_validation.json').write_text(json.dumps(rows, indent=2) + '\n')


def ratio_statistics(df):
    f = df.copy()
    f['sector_i'] = sector_index(f.wind_direction_100m)
    f['sector'] = [SECTORS[i] for i in f.sector_i]
    local = utc_to_local(f.index)
    f['year'] = local.year
    seasons = np.array(['winter', 'winter', 'spring', 'spring', 'spring', 'summer', 'summer', 'summer', 'autumn', 'autumn', 'autumn', 'winter'])
    f['season'] = seasons[local.month - 1]
    f['season_year'] = local.year + (local.month == 12)
    periods = [f'{year}_{season}' for year in sorted(f.season_year.unique())
               for season in ['winter', 'spring', 'summer', 'autumn']]
    f['period'] = pd.Categorical(f['season_year'].astype(str) + '_' + f['season'], categories=periods, ordered=True)
    # Exclude near-zero denominator; all coefficients are diagnostic only.
    f = f[f.wind_speed_100m >= 2].copy()
    f['ratio'] = f.wind_meas / f.wind_speed_100m
    f['speed_bin'] = pd.cut(f.wind_speed_100m, [2, 4, 6, 8, 10, 12, np.inf],
                             labels=['2–4', '4–6', '6–8', '8–10', '10–12', '12+'], right=False)
    dimensions = {'sector': ['sector_i', 'sector'], 'speed': ['sector_i', 'sector', 'speed_bin'],
                  'year': ['sector_i', 'sector', 'year'],
                  'season_year': ['sector_i', 'sector', 'season_year', 'season'],
                  'cube': ['sector_i', 'sector', 'season_year', 'season', 'speed_bin']}
    for name, extra in dimensions.items():
        table = f.groupby(['turbine', 'lead_day'] + extra, observed=True).agg(
            n=('ratio', 'size'), median_ratio=('ratio', 'median'),
            q25=('ratio', lambda x: x.quantile(.25)), q75=('ratio', lambda x: x.quantile(.75)),
            mean_power=('power', 'mean'), mean_forecast_wind=('wind_speed_100m', 'mean'),
            mean_measured_wind=('wind_meas', 'mean')).reset_index()
        table.to_csv(DATA / f'R1_ratio_{name}.csv', index=False)
    f['flow'] = np.where(f.sector_i.isin([3, 4]), 'ENE/E', np.where(f.sector_i.isin([11, 12]), 'WSW/W', 'other'))
    flow = f[f.flow != 'other'].groupby(['turbine', 'lead_day', 'flow', 'season_year', 'season', 'speed_bin'], observed=True).agg(
        n=('ratio', 'size'), median_ratio=('ratio', 'median'), mean_power=('power', 'mean')).reset_index()
    flow.to_csv(DATA / 'R1_flow_comparison.csv', index=False)
    return f


def comparisons(df):
    weather = df[['turbine', 'lead_day', 'wind_direction_100m']].reset_index()
    weather['time'] = pd.to_datetime(weather['time'], utc=True)
    rows, detail = [], []
    for name in VARIANTS:
        path = DATA / f'R1_{name}_predictions.csv'
        p = pd.read_csv(path, parse_dates=['time'])
        p['time'] = pd.to_datetime(p['time'], utc=True)
        # No index-only join: timestamps repeat across turbines and lead days.
        p = p.merge(weather, on=['time', 'turbine', 'lead_day'], how='left', validate='one_to_one')
        assert p.wind_direction_100m.notna().all()
        p['sector_i'] = sector_index(p.wind_direction_100m)
        p['sector'] = [SECTORS[i] for i in p.sector_i]
        p['error'] = p.p50 - p.power
        p['covered'] = (p.power >= p.p10) & (p.power <= p.p90)
        p['width'] = p.p90 - p.p10
        def record(g, **keys):
            return {'variant': name, **keys, 'n': len(g), 'mae': float(g.error.abs().mean()),
                    'rmse': float(np.sqrt((g.error**2).mean())), 'bias': float(g.error.mean()),
                    'coverage_pct': float(100 * g.covered.mean()), 'width': float(g.width.mean())}
        rows.append(record(p, group='overall'))
        for (t, lead), g in p.groupby(['turbine', 'lead_day']):
            rows.append(record(g, group=f'{t}_lead{lead}'))
        for (i, sector), g in p.groupby(['sector_i', 'sector']):
            rows.append(record(g, group=f'sector_{sector}', sector_i=int(i)))
        for (t, lead, i, sector), g in p.groupby(['turbine', 'lead_day', 'sector_i', 'sector']):
            detail.append(record(g, turbine=t, lead_day=int(lead), sector_i=int(i), sector=sector))
    table = pd.DataFrame(rows)
    table.to_csv(DATA / 'R1_comparison.csv', index=False)
    pd.DataFrame(detail).to_csv(DATA / 'R1_comparison_by_turbine_lead_sector.csv', index=False)
    return table


def figures(df, ratios, comparison):
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .2})
    # Map the actual OSM footprint over the checked-in DEM.
    lat0, lon0 = np.mean(list(config.TURBINES.values()), axis=0)
    grid = pd.read_csv(config.ROOT / 'data/terrain/dem_grid.csv')
    elements = json.loads((config.ROOT / 'data/terrain/osm_context.json').read_text())['elements']
    gx, gy = to_local(grid.lat, grid.lon, lat0, lon0)
    fig, ax = plt.subplots(figsize=(8, 7))
    heights = ax.tricontourf(gx / 1000, gy / 1000, grid.elevation_m, levels=18, cmap='terrain', alpha=.7)
    fig.colorbar(heights, ax=ax, label='Elevation, m', shrink=.7)
    for kind, color in [('residential', '#ad6856'), ('forest', '#246c3b')]:
        first = True
        for outer, inner in polygons(elements, kind):
            for ring in outer:
                ll = np.asarray(ring); x, y = to_local(ll[:, 1], ll[:, 0], lat0, lon0)
                ax.add_patch(Polygon(np.column_stack([x / 1000, y / 1000]), facecolor=color, alpha=.65,
                                     edgecolor=color, label=kind if first else None))
                first = False
    turbines = [e for e in elements if e.get('type') == 'node' and e.get('tags', {}).get('generator:source') == 'wind']
    xx, yy = to_local([e['lat'] for e in turbines], [e['lon'] for e in turbines], lat0, lon0)
    ax.scatter(xx / 1000, yy / 1000, marker='^', color='#163547', s=35, label='OSM turbines (15)', zorder=5)
    for tid, (lat, lon) in config.TURBINES.items():
        x, y = to_local(lat, lon, lat0, lon0)
        ax.scatter(x / 1000, y / 1000, color='#e64622', marker='*', s=120, zorder=6)
        ax.annotate(tid.upper(), (x / 1000, y / 1000), xytext=(10, -15 if tid == 't2' else 8), textcoords='offset points', weight='bold')
        if tid == 't1':
            for radius, style in [(1, ':'), (2, '--'), (3, ':')]:
                ax.add_patch(Circle((x / 1000, y / 1000), radius, fill=False, linestyle=style, color='#163547', linewidth=.9))
            angle = np.deg2rad([292.5 - 15, 292.5 + 15])
            for a in angle:
                ax.plot([x / 1000, x / 1000 + 2 * np.sin(a)], [y / 1000, y / 1000 + 2 * np.cos(a)], color='#24396c', linewidth=1.4)
    ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), aspect='equal', xlabel='East, km', ylabel='North, km',
           title='R1 · actual OSM layout and DEM\nWNW upstream cone ±15°, 2 km; rings 1 / 2 / 3 km')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / 'R1_site.png', dpi=160); plt.close(fig)
    # Ratios by sector/season and speed; do not show tiny cells as firm evidence.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, tid in enumerate(config.TURBINES):
        f = ratios[(ratios.turbine == tid) & (ratios.lead_day == 1)]
        grouped = f.groupby(['period', 'sector_i'], observed=True).ratio.agg(['median', 'size'])
        med = grouped['median'].where(grouped['size'] >= 30).unstack().reindex(columns=range(16))
        im = axes[row, 0].imshow(med.to_numpy(), aspect='auto', cmap='RdBu_r', vmin=.6, vmax=1.6)
        axes[row, 0].set_xticks(range(16), SECTORS, rotation=45)
        axes[row, 0].set_yticks(range(len(med)), med.index)
        axes[row, 0].set_title(f'{tid.upper()} · measured / forecast wind · day 1 · n ≥ 30')
        fig.colorbar(im, ax=axes[row, 0], shrink=.8)
        for flow, color in [('ENE/E', '#2879a3'), ('WSW/W', '#bd633b')]:
            s = f[f.flow == flow].groupby('speed_bin', observed=False).ratio.agg(['median', 'size'])
            axes[row, 1].plot(range(len(s)), s['median'], marker='o', color=color, label=flow)
            for i, value in enumerate(s['median']):
                axes[row, 1].annotate(str(int(s['size'].iloc[i])), (i, value), xytext=(0, 7 if flow == 'ENE/E' else -13),
                                     textcoords='offset points', ha='center', fontsize=7, color=color)
            axes[row, 1].set_xticks(range(len(s)), [str(x) for x in s.index])
        axes[row, 1].axhline(1, color='gray', linestyle='--', linewidth=.8)
        axes[row, 1].set(xlabel='Forecast wind at 100 m, m/s', ylabel='Median measured / forecast wind',
                          title=f'{tid.upper()} · direction × speed (labels = sample counts)')
        axes[row, 1].legend()
        axes[row, 1].margins(y=.14)
    fig.tight_layout(); fig.savefig(FIGURES / 'R1_wind_ratios.png', dpi=160); plt.close(fig)
    # Absolute errors by all 16 sectors plus the four contractual groups.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    labels = ['t1_lead1', 't1_lead2', 't2_lead1', 't2_lead2']
    for offset, name, title, color in [(-.23, 'baseline', 'Baseline', '#697c8c'), (0, 'all', 'All 6 proxies', '#bc8f4c'),
                                      (.23, 'selected', 'Selected: 2 slopes', '#238973')]:
        s = comparison[comparison.variant == name].set_index('group')
        bars = axes[0].bar(np.arange(4) + offset, s.loc[labels, 'mae'], .23, label=title, color=color)
        axes[0].bar_label(bars, fmt='%.3f', fontsize=7, padding=3)
        sector_rows = s.reindex(['sector_' + v for v in SECTORS])
        axes[1].plot(range(16), sector_rows.mae, marker='o', linewidth=1, markersize=3, label=title, color=color)
    axes[0].set_xticks(range(4), labels); axes[0].set(ylabel='MAE, rated power', ylim=(0, .215))
    axes[0].legend(loc='upper left', fontsize=8)
    axes[1].set_xticks(range(16), SECTORS, rotation=45); axes[1].set(ylabel='MAE, rated power')
    axes[1].legend(fontsize=8)
    counts = comparison[(comparison.variant == 'baseline') & comparison.group.str.startswith('sector_')].set_index('group').n
    axes[1].set_xlabel('Samples per sector: ' + ', '.join(str(int(counts.get('sector_' + s, 0))) for s in SECTORS), fontsize=7)
    fig.suptitle('R1 · December 2025–January 2026 · unchanged calibration protocol')
    fig.tight_layout(); fig.savefig(FIGURES / 'R1_backtest.png', dpi=160); plt.close(fig)


def context_table():
    rows = []
    for tid, rec in load_context().items():
        directions = np.arange(16) * 22.5
        count, weighted = wake_counts(directions, rec['positions'])
        for i, direction in enumerate(directions):
            rad = np.deg2rad(direction)
            rows.append({'turbine': tid, 'sector': SECTORS[i], 'direction_deg': direction,
                         'wake_upwind_count': int(count[i]), 'wake_distance_weight': float(weighted[i]),
                         'forest_fraction': rec['forest'][i], 'residential_fraction': rec['residential'][i],
                         'upwind_slope': rec['slope'][i],
                         'local_slope': -rec['gradient'][0] * np.sin(rad) - rec['gradient'][1] * np.cos(rad)})
    pd.DataFrame(rows).to_csv(DATA / 'R1_sector_context.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--backtests', action='store_true')
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    df, _, y = model.build_dataset(WeatherClient(offline=True))
    if args.validation:
        validation(df, all_features(df), y)
    if args.backtests:
        backtests()
    ratios = ratio_statistics(df)
    comparison = comparisons(df)
    context_table()
    figures(df, ratios, comparison)
    print(comparison[comparison.group == 'overall'].to_string(index=False))
