"""3D-сцена для любой площадки и выпуска: payload в контракте window.VIZ_DATA и самодостаточная страница (офлайн)."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from wind_agent import config, vizdata

FORECAST_CSV = config.OUTPUTS_DIR / "forecasts" / "2026-02-10.csv"


@pytest.fixture(scope="module")
def fc_nurly() -> pd.DataFrame:
    if not FORECAST_CSV.exists():
        pytest.skip("нет outputs/forecasts/2026-02-10.csv")
    return pd.read_csv(FORECAST_CSV)


def _fake_dem(lat0, lon0, out_dir, half_size_m=6000.0, step_m=250.0, pause_s=0.0, progress=None, turbines=None, **kw):
    """Синтетический DEM вместо Open-Meteo: наклонная плоскость + холм, те же файлы, что пишет fetch_dem_grid."""
    grid = vizdata.build_grid(lat0, lon0, half_size_m, step_m)
    grid["elevation_m"] = 400 + grid["x_m"] * 0.01 + 80 * np.exp(-(grid["x_m"] ** 2 + grid["y_m"] ** 2) / 4e6)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid[["lat", "lon", "elevation_m", "i", "j", "x_m", "y_m"]].to_csv(out_dir / "dem_grid.csv", index=False)
    n = int(grid["i"].max()) + 1
    (out_dir / "meta.json").write_text(json.dumps({"source": "синтетический DEM (тест)", "center": {"lat": lat0, "lon": lon0},
                                                   "half_size_m": half_size_m, "step_m": step_m, "nx": n, "ny": n}))
    if progress:
        progress("Рельеф: запрос 1/1")
    return grid


def test_nurly_payload_uses_panel_forecast(fc_nurly):
    p = vizdata.viz_payload(config.NURLY, fc_nurly, fetch=False)
    assert len(p["forecasts"]) == 1
    issue = p["forecasts"][0]
    assert issue["issue_date"] == "2026-02-10" and issue["n_hours"] == 48 and len(issue["hours"]) == 48
    h = issue["hours"][0]
    assert h["p50"] is not None and h["p50_t1"] is not None and h["p50_t2"] is not None
    assert {t["id"] for t in p["turbines"]} == {"t1", "t2"}
    assert p["terrain"]["nx"] * p["terrain"]["ny"] == len(p["terrain"]["elev"])
    assert p["site_name"] == config.NURLY.name and p["forecast_source"] == "panel"


def test_farm_computed_when_missing(fc_nurly):
    no_farm = fc_nurly[fc_nurly["turbine"] != "farm"]
    issue = vizdata.forecasts_payload(no_farm)[0]
    assert len(issue["hours"]) == 48
    h = issue["hours"][5]
    assert math.isclose(h["p50"], (h["p50_t1"] + h["p50_t2"]) / 2, abs_tol=2e-4)


def test_viz_html_inlines_data(fc_nurly):
    p = vizdata.viz_payload(config.NURLY, fc_nurly, fetch=False)
    htm = vizdata.viz_html(p, "2026-02-10")
    assert "window.VIZ_DATA = {" in htm and 'window.VIZ_DEFAULT_ISSUE = "2026-02-10"' in htm
    assert vizdata.VIZ_DATA_TAG not in htm
    assert 'id="wa-viz-main"' in htm and "three.js не загрузился" in htm


def test_custom_site_payload(fc_nurly, tmp_path, monkeypatch):
    site = config.custom_site(48.0, 68.0, n_turbines=7, rated_mw=3.0)
    monkeypatch.setattr(vizdata, "fetch_dem_grid", _fake_dem)
    fc = fc_nurly[fc_nurly["turbine"].isin(["t1", "farm"])].copy()
    fc["turbine"] = fc["turbine"].replace({"t1": "u1"})
    msgs: list[str] = []
    p = vizdata.viz_payload(site, fc, terrain_dir=tmp_path / site.key, fetch=True, progress=msgs.append)
    assert p["terrain"]["nx"] == p["terrain"]["ny"] == 25 and not p["terrain"]["synthetic"]
    ids = [t["id"] for t in p["turbines"]]
    assert len(ids) == 7 and ids[0] == "u1" and ids[-1] == "u7"
    # ряд поперёк преобладающего ветра, шаг 500 м
    t0, t1 = p["turbines"][0], p["turbines"][1]
    assert math.isclose(math.hypot(t1["x_m"] - t0["x_m"], t1["y_m"] - t0["y_m"]), 500, abs_tol=0.5)
    assert p["osm"]["available"] is False and p["wind_rose"]["all"]["n_hours"] == 48
    assert len(p["forecasts"]) == 1 and "p50_u1" in p["forecasts"][0]["hours"][0]
    assert any("Рельеф" in m for m in msgs)
    assert vizdata.viz_html(p, "2026-02-10").count("window.VIZ_DATA = ") == 1


def test_custom_site_without_dem_is_flat(tmp_path):
    site = config.custom_site(50.0, 70.0, n_turbines=2)
    fc = pd.DataFrame()
    p = vizdata.viz_payload(site, fc, terrain_dir=tmp_path / "none", fetch=False)
    assert p["terrain"]["synthetic"] and p["terrain"]["nx"] == 25 and len(p["turbines"]) == 2
    assert p["forecasts"] == [] and not vizdata.terrain_ready(site)
