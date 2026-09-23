"""Audit and validate exact-hour weather neighbors before any SCADA filtering."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from wind_agent import config
from wind_agent.data import utc_to_local
from wind_agent.model import HGB_PARAMS, metrics
from wind_agent.weather import WeatherClient
from research_forecast_improvement import OUT, FOLDS, legacy_dataset, point_metrics, fit_variant


def corrected_matrix(df, X, ensemble_context=False):
    corrected = X.copy()
    wx = WeatherClient(offline=True)
    for turbine in config.TURBINES:
        raw = wx.previous_runs(turbine, config.PREVIOUS_RUNS_START, config.HISTORY_END).set_index('time')
        members = {k: v.set_index('time') for k,v in wx.ensemble_previous_runs(turbine, config.PREVIOUS_RUNS_START, config.HISTORY_END).items()} if ensemble_context else {}
        for lead in config.LEAD_DAYS:
            mask = (df.turbine == turbine) & (df.lead_day == lead)
            times = df.index[mask]
            column = f'wind_speed_100m_previous_day{lead}'
            ws = raw[column]
            ensemble = pd.concat([ws] + [members[m][column].reindex(raw.index).fillna(ws) if m in members else ws for m in config.ENSEMBLE_MODELS], axis=1).mean(axis=1) if ensemble_context else None
            dates = utc_to_local(times).normalize()
            for name, offset in [('ws100_prev', -1), ('ws100_next', 1)]:
                neighbor = times + pd.Timedelta(hours=offset)
                same_day = (dates == utc_to_local(neighbor).normalize()) & (utc_to_local(neighbor) == utc_to_local(times) + pd.Timedelta(hours=offset))
                values = ws.reindex(neighbor).to_numpy()
                current = df.loc[mask, 'wind_speed_100m'].to_numpy()
                corrected.loc[mask, name] = np.where(same_day & np.isfinite(values), values, current)
                if ensemble_context:
                    values = ensemble.reindex(neighbor).to_numpy()
                    current = X.loc[mask, 'ens_mean_ws100'].to_numpy()
                    corrected.loc[mask, name.replace('ws100_', 'ens_ws100_')] = np.where(same_day & np.isfinite(values), values, current)
    return corrected


def validation(df, baseline, corrected, y, prefix='neighbors'):
    rows = []
    for cutoff in FOLDS:
        start = pd.Timestamp(cutoff, tz='UTC')
        train = df.index < start
        test = (df.index >= start) & (df.index < start + pd.DateOffset(months=2))
        for name, X in [('baseline', baseline), ('exact_weather_neighbors', corrected)]:
            m = HistGradientBoostingRegressor(loss='squared_error', **HGB_PARAMS).fit(X[train], y[train])
            row = {'fold': cutoff, 'variant': name, **point_metrics(df[test], m.predict(X[test]).clip(0,1))}
            rows.append(row)
            print(json.dumps({k:row[k] for k in ['fold','variant','mae','rmse','ramp_mae']}), flush=True)
    ratios = {k: [rows[i+1][k]/rows[i][k] for i in range(0,len(rows),2)] for k in ['mae','rmse','ramp_mae']}
    # Correct a train/inference mismatch if no material fold regression.
    accepted = (max(ratios['mae']) <= 1.02 and max(ratios['rmse']) <= 1.02 and max(ratios['ramp_mae']) <= 1.03
                and np.mean(ratios['mae']) <= 1 and np.mean(ratios['rmse']) <= 1.01)
    result = {'protocol': {'folds': FOLDS, 'source': 'previous_runs weather before SCADA filter',
               'boundary': 'same turbine, same lead, same nominal local-day release; exact +/-1h only; otherwise current wind',
               'gate': 'every fold MAE and RMSE <=102%, ramp MAE <=103%; mean MAE <=100%, mean RMSE <=101% baseline',
               'exploratory_after_previous_holdout_inspection': True, 'ensemble_context': 'ens_ws100_prev, ens_ws100_next' if prefix == 'neighbors_ensemble' else None},
              'rows': rows, 'relative_metrics': ratios, 'accepted': bool(accepted)}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{prefix}_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ACCEPTED',accepted,ratios,flush=True)
    return accepted


def final_comparison(df, baseline, corrected, y, prefix='neighbors'):
    start = pd.Timestamp(config.BACKTEST_TEST_START, tz='UTC')
    train = df.index < start
    cal = train & (df.index >= pd.Timestamp(config.BACKTEST_CALIB_START, tz='UTC'))
    test = (df.index >= start) & (df.index < pd.Timestamp(config.BACKTEST_TEST_END, tz='UTC')+pd.Timedelta(days=1))
    rows = {}
    for name, X in [('baseline',baseline), ('exact_weather_neighbors',corrected)]:
        c = fit_variant(X[train&~cal],y[train&~cal],'squared_error')
        m = fit_variant(X[train],y[train],'squared_error')
        m.calibration = c.calibrate(X[cal],y[cal])
        p=m.predict(X[test])
        g=df[test].copy()
        g[['p10','p50','p90']]=p
        row=point_metrics(g,p.p50.to_numpy())
        row.update(coverage_pct=float(100*g.power.between(g.p10,g.p90).mean()), width=float((g.p90-g.p10).mean()),calibration=m.calibration)
        row['sectors']={str(s):metrics(h.power,h.p50) for s,h in g.groupby(np.floor(((g.wind_direction_100m%360)+11.25)/22.5).astype(int)%16)}
        rows[name]=row
        g[['turbine','lead_day','power','p10','p50','p90','wind_speed_100m','wind_direction_100m']].to_csv(OUT/f'{prefix}_holdout_{name}.csv')
        print(name,{k:row[k] for k in ['mae','rmse','ramp_mae','coverage_pct','width']},flush=True)
    result={'exploratory':True,'selection_reason':'Correct weather semantics independently of the SCADA target filter; no claim of early accuracy improvement','records':rows}
    (OUT/f'{prefix}_holdout.json').write_text(json.dumps(result,indent=2)+'\n')


def full_training_validation(df, X, y):
    """One bounded follow-up: same 500 iterations, no random early-stop split."""
    rows = []
    for cutoff in FOLDS:
        start = pd.Timestamp(cutoff, tz='UTC')
        train = df.index < start
        test = (df.index >= start) & (df.index < start + pd.DateOffset(months=2))
        for name, extra in [('corrected_baseline', {}), ('all_training_rows', {'early_stopping': False})]:
            m = HistGradientBoostingRegressor(loss='squared_error', **{**HGB_PARAMS, **extra}).fit(X[train], y[train])
            row = {'fold': cutoff, 'variant': name, 'n_iter': int(m.n_iter_),
                   **point_metrics(df[test], m.predict(X[test]).clip(0, 1))}
            rows.append(row)
            print({k: row[k] for k in ['fold', 'variant', 'n_iter', 'mae', 'rmse', 'ramp_mae']}, flush=True)
    result = {'protocol': 'fixed500iterations; compare early_stopping=False only, no final test used', 'rows': rows}
    (OUT / 'full_training_validation.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--holdout',action='store_true')
    p.add_argument('--ensemble-context',action='store_true')
    p.add_argument('--all-training-rows', action='store_true')
    args=p.parse_args()
    begin=time.monotonic()
    df,X,y=legacy_dataset()
    corrected=corrected_matrix(df,X,args.ensemble_context)
    prefix='neighbors_ensemble' if args.ensemble_context else 'neighbors'
    groups=[df.turbine.to_numpy(),df.lead_day.to_numpy()]
    times=pd.Series(df.index,index=df.index)
    gap=times.groupby(groups).diff().gt(pd.Timedelta(hours=1))
    audit={'n':len(df),'nonadjacent_previous_rows':int(gap.sum()),'changed_previous':int((X.ws100_prev!=corrected.ws100_prev).sum()),'changed_next':int((X.ws100_next!=corrected.ws100_next).sum())}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'neighbors_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('AUDIT',audit,flush=True)
    if args.all_training_rows:
        full_training_validation(df, corrected, y)
    elif args.holdout:
        if args.ensemble_context and not json.loads((OUT/f'{prefix}_validation.json').read_text())['accepted']:
            raise SystemExit('Ensemble context failed the early gate; do not select it by the final test.')
        # The exact-hour correction is mandatory data semantics, not an accuracy winner.
        # Its already-opened final-test result is reported transparently, never used to tune.
        final_comparison(df,X,corrected,y,prefix)
    else:
        validation(df,X,corrected,y,prefix)
    print('SECONDS',round(time.monotonic()-begin,2),flush=True)


if __name__=='__main__':
    main()
