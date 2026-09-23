"""Offline directional terrain proxies from the checked-in OSM and DEM snapshots.

Same local projection as scripts/build_viz_data.py: x east, y north, metres.
Wind direction is the bearing FROM which wind arrives. These are geometric
proxies, not a CFD/Jensen solution or a measured aerodynamic roughness length.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from matplotlib.path import Path as PolygonPath

from . import config

log = logging.getLogger(__name__)
TERRAIN_DIR = config.ROOT / 'data' / 'terrain'
TERRAIN_FEATURES = [
    'wake_upwind_count', 'wake_distance_weight',
    'upwind_forest_fraction', 'upwind_residential_fraction',
    'terrain_upwind_slope', 'terrain_local_slope',
]
SECTOR_WIDTH = 22.5
RASTER_STEP_M = 50.0


def to_local(lat, lon, lat0: float, lon0: float):
    """Projection shared with the existing terrain visualisation (within 6 km)."""
    return ((np.asarray(lon) - lon0) * 111320 * np.cos(np.deg2rad(lat0)),
            (np.asarray(lat) - lat0) * 111132)


def sector_index(direction):
    return np.floor(((np.asarray(direction) % 360) + SECTOR_WIDTH / 2) / SECTOR_WIDTH).astype(int) % 16


def bearing(x, y):
    return np.rad2deg(np.arctan2(x, y)) % 360


def wake_counts(direction, positions: np.ndarray):
    """Other rotors within 2 km and +/-15 degrees; positions are relative xy metres.

    Exclude coordinates within 20 m of this turbine (OSM/SCADA position tolerance).
    The other modelled turbine remains eligible; only this rotor is excluded.
    """
    direction = np.asarray(direction, dtype=float)
    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    dist = np.linalg.norm(positions, axis=1)
    use = (dist > 20) & (dist <= 2000)
    dist, positions = dist[use], positions[use]
    angles = bearing(positions[:, 0], positions[:, 1])
    delta = ((direction[:, None] - angles + 180) % 360) - 180
    inside = np.abs(delta) <= 15 + 1e-10
    return inside.sum(axis=1), (inside / (1 + dist / 500)).sum(axis=1)


def assemble_rings(parts):
    """Join reversed way fragments as in build_viz_data._assemble_rings.

    Only closed rings are accepted: open OSM ways must not acquire a made-up
    closing boundary. Coordinates here stay in lon/lat until projection.
    """
    parts = [list(p) for p in parts if len(p) >= 2]
    rings = []
    while parts:
        ring = parts.pop(0)
        while ring[0] != ring[-1]:
            for k, p in enumerate(parts):
                if p[0] == ring[-1]:
                    ring += p[1:]
                elif p[-1] == ring[-1]:
                    ring += p[::-1][1:]
                elif p[-1] == ring[0]:
                    ring = p + ring[1:]
                elif p[0] == ring[0]:
                    ring = p[::-1] + ring[1:]
                else:
                    continue
                parts.pop(k)
                break
            else:
                break
        if len(ring) >= 4 and ring[0] == ring[-1]:
            rings.append(ring)
    return rings


def polygons(elements, kind):
    """Return (outer rings, holes), respecting multipolygon member roles."""
    out = []
    def land_kind(el):
        tags = el.get('tags', {})
        return ('forest' if tags.get('landuse') == 'forest' or tags.get('natural') == 'wood'
                else 'residential' if tags.get('landuse') == 'residential' else None)

    relations = [e for e in elements if e.get('type') == 'relation' and land_kind(e) == kind]
    member_ids = {m.get('ref') for e in relations for m in e.get('members', []) if m.get('type') == 'way'}
    for el in elements:
        if land_kind(el) != kind:
            continue
        if el.get('type') == 'way' and el.get('id') not in member_ids:
            outer = assemble_rings([[(p['lon'], p['lat']) for p in el.get('geometry', [])]])
            inner = []
        elif el.get('type') == 'relation':
            def rings(role):
                return assemble_rings([[(p['lon'], p['lat']) for p in m['geometry']]
                                       for m in el.get('members', []) if m.get('type') == 'way'
                                       and m.get('geometry') and (m.get('role') or 'outer') == role])
            outer, inner = rings('outer'), rings('inner')
        else:
            continue
        if outer:
            out.append((outer, inner))
    return out


def land_fraction(elements, kind, lat0, lon0):
    """Area fraction by 16 sectors in the 1–3 km annulus, sampled on a 50 m grid.

    Boolean union avoids double-counting overlapping ways/relations; holes are
    subtracted within each polygon before union. No mapped polygon means zero.
    """
    axis = np.arange(-3000 + RASTER_STEP_M / 2, 3000, RASTER_STEP_M)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    dist = np.linalg.norm(points, axis=1)
    points = points[(dist >= 1000) & (dist <= 3000)]
    sectors = sector_index(bearing(points[:, 0], points[:, 1]))
    occupied = np.zeros(len(points), dtype=bool)
    for outer, inner in polygons(elements, kind):
        def contains(rings):
            mask = np.zeros(len(points), dtype=bool)
            for ring in rings:
                ll = np.asarray(ring)
                x, y = to_local(ll[:, 1], ll[:, 0], lat0, lon0)
                mask |= PolygonPath(np.column_stack([x, y])).contains_points(points)
            return mask
        occupied |= contains(outer) & ~contains(inner)
    return np.bincount(sectors, weights=occupied, minlength=16) / np.bincount(sectors, minlength=16)


def terrain_slopes(grid, lat0, lon0):
    """Local plane within 750 m; mean uphill slope from 1–3 km upwind to site.

    Positive slope rises along the arriving flow. Local along-flow slope is
    -gradient dot (sin(direction), cos(direction)), computed at inference time.
    """
    x, y = to_local(grid.lat.to_numpy(), grid.lon.to_numpy(), lat0, lon0)
    z = grid.elevation_m.to_numpy(dtype=float)
    dist = np.hypot(x, y)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    near = valid & (dist <= 750)
    A = np.column_stack([x[near], y[near], np.ones(near.sum())])
    if len(A) < 3 or np.linalg.matrix_rank(A) < 3:
        raise ValueError('DEM has insufficient local points for a terrain plane')
    a, b, site_z = np.linalg.lstsq(A, z[near], rcond=None)[0]
    annulus = valid & (dist >= 1000) & (dist <= 3000)
    sectors = sector_index(bearing(x[annulus], y[annulus]))
    total = np.bincount(sectors, weights=(site_z - z[annulus]) / dist[annulus], minlength=16)
    counts = np.bincount(sectors, minlength=16)
    slopes = np.divide(total, counts, out=np.zeros(16), where=counts > 0)
    return slopes, np.array([a, b])


@lru_cache(maxsize=1)
def load_context():
    """Read each static snapshot once per process; independent zero fallbacks."""
    try:
        elements = json.loads((TERRAIN_DIR / 'osm_context.json').read_text())['elements']
        if not isinstance(elements, list) or not all(
            isinstance(e, dict) and isinstance(e.get('tags', {}), dict) for e in elements
        ):
            raise ValueError('OSM elements must be a list of objects with object tags')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning('OSM terrain features unavailable; using zeros: %s', exc)
        elements = []
    try:
        grid = pd.read_csv(TERRAIN_DIR / 'dem_grid.csv')
        grid = grid[['lat', 'lon', 'elevation_m']].astype(float)
    except (OSError, ValueError, KeyError) as exc:
        log.warning('DEM terrain features unavailable; using zeros: %s', exc)
        grid = None
    result = {}
    for tid, (lat, lon) in config.TURBINES.items():
        rec = {'positions': np.empty((0, 2)), 'forest': np.zeros(16), 'residential': np.zeros(16),
               'slope': np.zeros(16), 'gradient': np.zeros(2)}
        try:
            nodes = {e['id']: e for e in elements if e.get('type') == 'node'
                     and e.get('tags', {}).get('power') == 'generator'
                     and e.get('tags', {}).get('generator:source') == 'wind'}
            xy = [to_local(e['lat'], e['lon'], lat, lon) for e in nodes.values()]
            rec['positions'] = np.asarray(xy, dtype=float).reshape(-1, 2)
            for kind in ['forest', 'residential']:
                rec[kind] = land_fraction(elements, kind, lat, lon)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            log.warning('Invalid OSM geometry for %s; unavailable features remain zero: %s', tid, exc)
        if grid is not None:
            try:
                rec['slope'], rec['gradient'] = terrain_slopes(grid, lat, lon)
            except (ValueError, np.linalg.LinAlgError) as exc:
                log.warning('Invalid DEM for %s; using zero slopes: %s', tid, exc)
        result[tid] = rec
    return result


def directional_features(df: pd.DataFrame) -> pd.DataFrame:
    values = np.zeros((len(df), len(TERRAIN_FEATURES)))
    directions = df['wind_direction_100m'].to_numpy(dtype=float)
    turbines = df['turbine'].to_numpy() if 'turbine' in df else np.repeat(next(iter(config.TURBINES)), len(df))
    for tid, rec in load_context().items():
        mask = (turbines == tid) & np.isfinite(directions)
        d = directions[mask] % 360
        sector = sector_index(d)
        count, weighted = wake_counts(d, rec['positions'])
        rad = np.deg2rad(d)
        a, b = rec['gradient']
        values[mask] = np.column_stack([count, weighted, rec['forest'][sector], rec['residential'][sector],
                                       rec['slope'][sector], -a * np.sin(rad) - b * np.cos(rad)])
    return pd.DataFrame(values, index=df.index, columns=TERRAIN_FEATURES)
