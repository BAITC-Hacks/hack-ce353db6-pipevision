"""Статичная карта площадки ВЭС для README (matplotlib):

  viz/site_map.png       — рельеф (заливка по высоте, изолинии через 20 м, отмывка), турбины парка (OSM),
                           наши T1/T2, застройка, две главные оси ветра из розы, масштаб, север;
  viz/site_map_wake.png  — та же карта + конусы ±15° до 2 км от T1 навстречу ветру 72° и 274°.

Данные: data/terrain/dem_grid.csv (scripts/fetch_dem.py), data/terrain/osm_context.json
(scripts/fetch_osm_context.py), роза ветров — из кэша Open-Meteo (как в scripts/build_viz_data.py).

Запуск:  .venv/bin/python scripts/make_site_map.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Wedge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_viz_data import SECTOR_NAMES, build_wind_rose, load_osm, load_terrain  # noqa: E402  (те же загрузчики)

OUT_MAP = ROOT / "viz" / "site_map.png"
OUT_WAKE = ROOT / "viz" / "site_map_wake.png"
DPI = 130
VIEW_KM = 4.0                      # показываем квадрат ±4 км (весь DEM — ±6 км)
CONE_HALF_DEG, CONE_RANGE_KM = 15.0, 2.0
HALO = [pe.withStroke(linewidth=3, foreground="white")]
CMAP = LinearSegmentedColormap.from_list("hypso", ["#5d8c58", "#9db77a", "#d8d196", "#c9a77a", "#f3efe7"])
LABEL_XY = {72.0: (1.9, -0.8), 274.0: (-2.3, -2.1)}   # где подписывать конусы (км), чтобы не перекрывать слои
CAPTION = "Высоты: Copernicus DEM (Open-Meteo); турбины и застройка: © OpenStreetMap"
# две главные оси розы: какие румбы суммируем (центры в градусах «откуда дует»)
LOBES = [("ВСВ", 67.5, ["СВ", "ВСВ", "В"]), ("ЗЮЗ–З", 258.75, ["ЮЗ", "ЗЮЗ", "З", "ЗСЗ"])]


def upsample(a: np.ndarray, k: int) -> np.ndarray:
    """Билинейное увеличение сетки в k раз (только numpy) — для гладкой отмывки."""
    ny, nx = a.shape
    xi, yi = np.linspace(0, nx - 1, (nx - 1) * k + 1), np.linspace(0, ny - 1, (ny - 1) * k + 1)
    tmp = np.array([np.interp(xi, np.arange(nx), row) for row in a])
    return np.array([np.interp(yi, np.arange(ny), col) for col in tmp.T]).T


def bearing_deg(dx: float, dy: float) -> float:
    return (math.degrees(math.atan2(dx, dy)) + 360) % 360


def in_cone(src: dict, pts: list[dict], wd: float) -> list[tuple[dict, float]]:
    """Турбины в конусе ±15° навстречу ветру wd (откуда дует) до 2 км — та же логика, что на странице."""
    hits = []
    for p in pts:
        dx, dy = p["x_m"] - src["x_m"], p["y_m"] - src["y_m"]
        d = math.hypot(dx, dy)
        if d < 1 or d > CONE_RANGE_KM * 1000:
            continue
        diff = abs((bearing_deg(dx, dy) - wd + 540) % 360 - 180)
        if diff <= CONE_HALF_DEG:
            hits.append((p, d))
    return sorted(hits, key=lambda h: h[1])


def draw_base(ax, terrain: dict, turbines: list[dict], osm: dict, rose: dict):
    nx, ny, st = terrain["nx"], terrain["ny"], terrain["step_m"]
    elev = np.array(terrain["elev"], dtype=float).reshape(ny, nx)          # строки: юг → север
    x_km = (terrain["x0_m"] + np.arange(nx) * st) / 1000
    y_km = (terrain["y0_m"] + np.arange(ny) * st) / 1000
    view = (np.abs(x_km)[None, :] <= VIEW_KM) & (np.abs(y_km)[:, None] <= VIEW_KM)
    vmin, vmax = float(elev[view].min()), float(elev[view].max())

    k = 4
    up = upsample(elev, k)
    ls = LightSource(azdeg=315, altdeg=40)
    rgb = ls.shade(np.flipud(up), cmap=CMAP, norm=Normalize(vmin, vmax), vert_exag=4,
                   dx=st / k, dy=st / k, blend_mode="soft", fraction=0.8)          # строка 0 = север
    ext = (x_km[0], x_km[-1], y_km[0], y_km[-1])
    ax.imshow(rgb, extent=ext, origin="upper", interpolation="bilinear", zorder=0)

    levels = np.arange(math.floor(vmin / 20) * 20, vmax + 20, 20)
    cs = ax.contour(x_km, y_km, elev, levels=levels, colors="#4b3f33", linewidths=0.55, alpha=0.55, zorder=1)
    lab = [lv for lv in levels if lv % 40 == 0]
    ax.clabel(cs, levels=lab, fmt=lambda v: f"{v:.0f}", fontsize=7, inline=True, inline_spacing=2)

    # застройка
    for poly in osm.get("residential", []):
        for ring in poly["rings"]:
            ax.add_patch(Polygon(np.array(ring) / 1000, closed=True, facecolor=(0.55, 0.42, 0.30, 0.30),
                                 edgecolor="#6b4f36", linewidth=1.4, zorder=3))
        lx, ly = np.array(poly["label_xy"]) / 1000
        if abs(lx) < VIEW_KM and abs(ly) < VIEW_KM:
            ax.text(lx, ly, poly["name"] or "застройка", fontsize=9 if poly["name"] else 7.5, color="#4a3322",
                    ha="center", va="center", fontweight="bold" if poly["name"] else "normal", path_effects=HALO, zorder=4)

    # турбины парка (OSM) и наши
    park = [t for t in osm.get("turbines", []) if not t["ours"]]
    if park:
        px, py = np.array([t["x_m"] for t in park]) / 1000, np.array([t["y_m"] for t in park]) / 1000
        ax.scatter(px, py, s=26, c="#6b7280", edgecolors="white", linewidths=0.7, zorder=6,
                   label=f"турбины парка (OSM): {len(park)}")
        ax.text(px.mean() - 0.15, py.min() - 0.22, f"ВЭС «Нурлы»\n{len(park)} турбин (OSM)", fontsize=8.5, ha="center",
                va="top", color="#374151", path_effects=HALO, zorder=7)
    model = turbines[0].get("model")
    ax.scatter([t["x_m"] / 1000 for t in turbines], [t["y_m"] / 1000 for t in turbines], s=70, c="#e4572e",
               edgecolors="white", linewidths=1.2, zorder=8, label="T1, T2 — наши" + (f" ({model})" if model else ""))
    for t, off in zip(turbines, [(7, 7), (9, -13)]):          # подписи вправо — к западу стоят турбины парка
        ax.annotate(t["name"], (t["x_m"] / 1000, t["y_m"] / 1000), xytext=off, textcoords="offset points",
                    fontsize=10, fontweight="bold", color="#b23a1b", path_effects=HALO, zorder=9,
                    ha="right" if off[0] < 0 else "left")

    # две главные оси ветра из розы (весь период)
    sp = dict(zip(SECTOR_NAMES, rose["all"]["sector_pct"]))
    for name, brg, sectors in LOBES:
        share = sum(sp[s] for s in sectors)
        b = math.radians(brg)
        start, end = (3.55 * math.sin(b), 3.55 * math.cos(b)), (2.35 * math.sin(b), 2.35 * math.cos(b))
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="simple,head_width=1.4,head_length=1.2,tail_width=0.55",
                                     mutation_scale=16, color="#1d4e89", alpha=0.85, zorder=5))
        tx, ty = 2.95 * math.sin(b), 2.95 * math.cos(b) + (0.5 if brg < 180 else -0.5)
        ax.text(tx, ty, f"{name}: {share:.1f} % часов".replace(".", ",") + f"\n({'+'.join(sectors)})", fontsize=8.5, color="#12365f",
                ha="center", va="center", path_effects=HALO, zorder=6)

    # масштаб и север
    x0, y0 = -3.75, -3.7
    ax.plot([x0, x0 + 1], [y0, y0], color="black", lw=2.5, solid_capstyle="butt", zorder=9)
    for xx in (x0, x0 + 0.5, x0 + 1):
        ax.plot([xx, xx], [y0 - 0.05, y0 + 0.05], color="black", lw=1.2, zorder=9)
    ax.text(x0 + 0.5, y0 + 0.12, "1 км", ha="center", va="bottom", fontsize=8.5, path_effects=HALO, zorder=9)
    ax.annotate("С", xy=(3.62, 3.72), xytext=(3.62, 3.05), ha="center", va="top", fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6, mutation_scale=14), path_effects=HALO, zorder=9)

    ax.set_xlim(-VIEW_KM, VIEW_KM)
    ax.set_ylim(-VIEW_KM, VIEW_KM)
    ax.set_aspect("equal")
    ax.set_xlabel("км на восток от центра площадки")
    ax.set_ylabel("км на север от центра площадки")
    ax.tick_params(labelsize=8)
    return vmin, vmax


def figure(terrain, turbines, osm, rose, title: str):
    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    vmin, vmax = draw_base(ax, terrain, turbines, osm, rose)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Normalize(vmin, vmax))
    fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.02, label="Высота, м")
    ax.set_title(title, fontsize=11)
    ax.text(0.5, -0.095, CAPTION, transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#444")
    return fig, ax


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.15, pil_kwargs={"optimize": True})
    plt.close(fig)
    kb = path.stat().st_size / 1024
    if kb > 500:  # на всякий случай ужимаем палитрой (Pillow идёт вместе с matplotlib)
        from PIL import Image
        Image.open(path).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
        kb = path.stat().st_size / 1024
    print(f"  {path.relative_to(ROOT)}: {kb:.0f} КБ")


def main() -> int:
    terrain, turbines = load_terrain()
    if terrain.get("synthetic"):
        print("ОШИБКА: нет DEM — сначала .venv/bin/python scripts/fetch_dem.py", file=sys.stderr)
        return 1
    osm = load_osm(terrain["center"]["lat"], terrain["center"]["lon"], turbines)
    rose = build_wind_rose()
    period = rose["all"]["label"].replace("Весь период ", "")

    fig, ax = figure(terrain, turbines, osm, rose,
                     f"Площадка ВЭС: рельеф, турбины и главные оси ветра на 100 м\n"
                     f"(доли часов по прогнозам Open-Meteo, {period})")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    save(fig, OUT_MAP)

    # ---- карта следа: конусы от T1
    fig, ax = figure(terrain, turbines, osm, rose,
                     "Наветренный сектор T1: конусы ±15° до 2 км\nС запада T1/T2 в следе парка: "
                     "замер/прогноз 0,95 против 1,25 с востока")
    t1 = next(t for t in turbines if t["id"] == "t1")
    targets = [t for t in osm.get("turbines", []) if t["ours"] != "t1"]
    for wd, lab_name in [(72.0, "ВСВ 72°"), (274.0, "З 274°")]:
        hits = in_cone(t1, targets, wd)
        color = "#f28e2b" if hits else "#2ca25f"
        ax.add_patch(Wedge((t1["x_m"] / 1000, t1["y_m"] / 1000), CONE_RANGE_KM, 90 - wd - CONE_HALF_DEG,
                           90 - wd + CONE_HALF_DEG, facecolor=color, alpha=0.28, edgecolor=color, linewidth=1.5, zorder=4))
        if hits:
            ax.scatter([h[0]["x_m"] / 1000 for h in hits], [h[0]["y_m"] / 1000 for h in hits], s=150,
                       facecolors="none", edgecolors="#c2410c", linewidths=2, zorder=7)
            txt = f"ветер {lab_name}: в конусе {len(hits)} турб. парка,\nближайшая {hits[0][1]:.0f} м"
        else:
            txt = f"ветер {lab_name}:\nнаветренная сторона свободна"
        ax.plot([t1["x_m"] / 1000 + 1.2 * math.sin(math.radians(wd)), LABEL_XY[wd][0]],
                [t1["y_m"] / 1000 + 1.2 * math.cos(math.radians(wd)), LABEL_XY[wd][1]], color=color, lw=1, zorder=7)
        ax.text(*LABEL_XY[wd], txt, fontsize=8.5, ha="center", va="center", color="#7c2d12" if hits else "#14532d",
                path_effects=HALO, zorder=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85))
        print(f"  конус {lab_name}: турбин в конусе {len(hits)}" + (f", ближайшая {hits[0][1]:.0f} м" if hits else ""))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    save(fig, OUT_WAKE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
