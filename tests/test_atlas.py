"""Атлас ветра Казахстана: маска границы, ближайшая ячейка, классы ресурса, GeoJSON ячеек (офлайн, по CSV из репо)."""
from __future__ import annotations

import numpy as np

from wind_agent import atlas, config


def test_polygon_mask():
    lat, lon = config.TURBINES["t1"]
    assert atlas.in_kazakhstan(lat, lon)                 # ВЭС «Нурлы»
    assert atlas.in_kazakhstan(51.17, 71.43)             # Астана
    assert not atlas.in_kazakhstan(41.3, 69.3)           # Ташкент
    assert not atlas.in_kazakhstan(42.5, 50.5)           # Каспийское море
    assert not atlas.in_kazakhstan(55.75, 37.6)          # Москва


def test_resource_class():
    assert atlas.resource_class(4.0) == "низкий"
    assert atlas.resource_class(5.5) == "умеренный"
    assert atlas.resource_class(7.0) == "хороший"
    assert atlas.resource_class(7.5) == "отличный"
    assert atlas.resource_class(float("nan")) == "нет данных"


def test_atlas_csv_and_nearest_cell():
    df = atlas.load_atlas()
    assert len(df) > 800 and df["inside_kz"].all()
    assert set(atlas.COLUMNS) <= set(df.columns)
    assert df["ws100_est"].between(1, 15).all()
    lat, lon = config.TURBINES["t1"]
    cell = atlas.nearest_cell(lat, lon)
    assert cell["distance_km"] < 40
    assert 3 < cell["ws100_est"] < 9
    assert cell["resource_class"] == atlas.resource_class(cell["ws100_est"])
    # ws100 — степенной закон от ws50
    assert np.isclose(cell["ws100_est"], cell["ws50_ann"] * 2 ** atlas.SHEAR_ALPHA, atol=0.01)


def test_grid_cells_geojson():
    gj = atlas.grid_cells_geojson()
    assert gj["type"] == "FeatureCollection"
    feats = gj["features"]
    assert len(feats) == len(atlas.load_atlas())
    f = feats[0]
    ring = f["geometry"]["coordinates"][0]
    assert f["geometry"]["type"] == "Polygon" and len(ring) == 5 and ring[0] == ring[-1]
    p = f["properties"]
    assert f["id"] == f"{p['lat']}_{p['lon']}"
    assert {"lat", "lon", "ws100_est", "ws50_ann", "resource_class"} <= set(p)
    xs, ys = [x for x, _ in ring], [y for _, y in ring]
    assert min(xs) < p["lon"] < max(xs) and min(ys) < p["lat"] < max(ys)     # порядок [lon, lat]
    assert len({x["id"] for x in feats}) == len(feats)
