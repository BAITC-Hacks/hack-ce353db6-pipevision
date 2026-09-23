"""Directional geometry, ring holes and offline degradation of R1 features."""
import json

import numpy as np
import pandas as pd
import pytest

from wind_agent import config, terrain
from wind_agent.features import MODEL_TERRAIN_FEATURES, make_features
from conftest import model_input


def test_from_bearing_wrap_radius_and_exclude_self():
    positions = np.array([[0, 0], [0, 10], [0, 1000], [0, 2000], [0, 2001], [500, 0]])
    count, weight = terrain.wake_counts([0, 360, 350, 15, 16, 90, 180], positions)
    np.testing.assert_array_equal(count, [2, 2, 2, 2, 0, 1, 0])
    assert weight[0] == pytest.approx(1 / 3 + 1 / 5)
    assert weight[5] == pytest.approx(.5)
    np.testing.assert_array_equal(terrain.sector_index([0, 359, 11.24, 11.25, 348.75]), [0, 0, 0, 1, 0])


def points(coords):
    return [{'lon': x / 111320, 'lat': y / 111132} for x, y in coords]


def test_land_union_and_multipolygon_hole():
    square = [(0, 0), (3500, 0), (3500, 3500), (0, 3500), (0, 0)]
    way = {'id': 1, 'type': 'way', 'tags': {'landuse': 'forest'}, 'geometry': points(square)}
    frac = terrain.land_fraction([way, {**way, 'id': 2}], 'forest', 0, 0)
    assert frac[2] == 1  # north-east; overlap is counted once
    assert frac[10] == 0
    assert ((frac >= 0) & (frac <= 1)).all()
    hole = [(500, 500), (3000, 500), (3000, 3000), (500, 3000), (500, 500)]
    relation = {'type': 'relation', 'id': 10, 'tags': {'natural': 'wood'}, 'members': [
        {'type': 'way', 'ref': 1, 'role': 'outer', 'geometry': points(square[:3])},
        {'type': 'way', 'ref': 2, 'role': 'outer', 'geometry': points(square[2:][::-1])},
        {'type': 'way', 'ref': 3, 'role': 'inner', 'geometry': points(hole)},
    ]}
    with_hole = terrain.land_fraction([relation, way], 'forest', 0, 0)
    assert with_hole[2] == 0  # member way must not refill the relation's hole
    assert terrain.assemble_rings([[(0, 0), (1, 0), (1, 1)]]) == []


def test_plane_exposure_sign():
    xx, yy = np.meshgrid(np.arange(-3000, 3001, 250), np.arange(-3000, 3001, 250))
    grid = pd.DataFrame({'lon': xx.ravel() / 111320, 'lat': yy.ravel() / 111132,
                         'elevation_m': 100 + .01 * xx.ravel() + .02 * yy.ravel()})
    slope, grad = terrain.terrain_slopes(grid, 0, 0)
    np.testing.assert_allclose(grad, [.01, .02], atol=1e-12)
    assert slope[0] == pytest.approx(-.02, abs=.001)
    assert slope[8] == pytest.approx(.02, abs=.001)
    assert slope[4] == pytest.approx(-.01, abs=.001)


@pytest.fixture
def temporary_context(monkeypatch, tmp_path):
    monkeypatch.setattr(terrain, 'TERRAIN_DIR', tmp_path)
    terrain.load_context.cache_clear()
    yield tmp_path
    terrain.load_context.cache_clear()


def test_missing_files_zero_features_without_changing_existing_inputs(temporary_context, archived_t1):
    X = make_features(model_input(archived_t1, 't1'))
    assert (X[MODEL_TERRAIN_FEATURES] == 0).all().all()
    assert (terrain.directional_features(model_input(archived_t1, 't1')) == 0).all().all()
    assert X['wind_speed_100m'].equals(archived_t1.set_index('time').wind_speed_100m)


def test_osm_available_when_dem_missing_and_duplicate_nodes(temporary_context):
    lat, lon = config.TURBINES['t1']
    node = {'type': 'node', 'id': 42, 'lat': lat + 1000 / 111132, 'lon': lon,
            'tags': {'power': 'generator', 'generator:source': 'wind'}}
    (temporary_context / 'osm_context.json').write_text(json.dumps({'elements': [node, node]}))
    idx = pd.DatetimeIndex(['2026-01-01'] * 4, tz='UTC')
    df = pd.DataFrame({'turbine': ['t1', 't1', 'missing', 't1'],
                       'wind_direction_100m': [0., 180., 0., np.nan]}, index=idx)
    x = terrain.directional_features(df)
    np.testing.assert_array_equal(x.wake_upwind_count, [1, 0, 0, 0])
    assert (x[['terrain_upwind_slope', 'terrain_local_slope']] == 0).all().all()
    assert np.isfinite(x).all().all()
    assert x.index.equals(idx)


@pytest.mark.parametrize('payload', ['{invalid', '{"elements": null}', '{"elements": [null]}',
                                    '{"elements": [{"tags": null}]}'])
def test_corrupt_files_zero_fallback(temporary_context, payload):
    (temporary_context / 'osm_context.json').write_text(payload)
    (temporary_context / 'dem_grid.csv').write_text('wrong,column\n1,2\n')
    df = pd.DataFrame({'turbine': ['t1'], 'wind_direction_100m': [90.]})
    assert (terrain.directional_features(df) == 0).all().all()


def test_checked_in_context_has_both_rotors_and_no_mapped_forest():
    ctx = terrain.load_context()
    for rec in ctx.values():
        assert len(rec['positions']) == 15
        assert np.linalg.norm(rec['positions'], axis=1).min() < 20
        assert rec['forest'].max() == 0  # missing OSM polygons, not proof there is no forest
        assert rec['residential'].max() > .4


def test_dem_available_when_osm_missing(temporary_context):
    lat, lon = config.TURBINES['t1']
    xx, yy = np.meshgrid(np.arange(-3000, 3001, 250), np.arange(-3000, 3001, 250))
    pd.DataFrame({'lon': lon + xx.ravel() / (111320 * np.cos(np.deg2rad(lat))),
                  'lat': lat + yy.ravel() / 111132,
                  'elevation_m': 100 + .01 * xx.ravel() + .02 * yy.ravel()}).to_csv(temporary_context / 'dem_grid.csv', index=False)
    df = pd.DataFrame({'turbine': ['t1'], 'wind_direction_100m': [0.]})
    x = terrain.directional_features(df)
    assert (x[terrain.TERRAIN_FEATURES[:4]] == 0).all().all()
    assert x.terrain_local_slope.iloc[0] == pytest.approx(-.02, abs=1e-9)
    assert x.terrain_upwind_slope.iloc[0] == pytest.approx(-.02, abs=.001)
