"""Контекст местности из OpenStreetMap для 3D-визуализации: турбины парка, леса, застройка.

Overpass API, радиус 5 км от центра площадки (середина между T1 и T2):
  * node["power"="generator"]["generator:source"="wind"]      — ветроустановки
  * way["landuse"="forest"], way["natural"="wood"]             — лес (+ мультиполигоны relation)
  * way["landuse"="residential"]                               — застройка (посёлок Нурлы и др.)
Геометрия запрашивается через `out geom`. Ответ сохраняется как есть (elements) в data/terrain/osm_context.json.

Скрипт идемпотентный: если файл уже есть — ничего не качает (перекачать — --force).
Если Overpass недоступен — понятная ошибка и код 1; страница при этом работает без слоёв OSM.

Запуск:  .venv/bin/python scripts/fetch_osm_context.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:  # координаты турбин из конфига проекта (только чтение)
    from wind_agent.config import TURBINES
except Exception:  # pragma: no cover
    TURBINES = {"t1": (43.645150, 78.535604), "t2": (43.643198, 78.538828)}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OUT_JSON = ROOT / "data" / "terrain" / "osm_context.json"
RADIUS_M = 5000
TIMEOUT_S = 90

QUERY = """[out:json][timeout:60];
(
  node["power"="generator"]["generator:source"="wind"](around:{r},{lat},{lon});
  way["landuse"="forest"](around:{r},{lat},{lon});
  way["natural"="wood"](around:{r},{lat},{lon});
  relation["landuse"="forest"](around:{r},{lat},{lon});
  relation["natural"="wood"](around:{r},{lat},{lon});
  way["landuse"="residential"](around:{r},{lat},{lon});
  relation["landuse"="residential"](around:{r},{lat},{lon});
);
out geom;"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Выгрузить турбины, леса и застройку из OSM (Overpass API)")
    ap.add_argument("--force", action="store_true", help="перекачать, даже если файл уже есть")
    ap.add_argument("--url", default=OVERPASS_URL, help="адрес Overpass API (по умолчанию %(default)s)")
    args = ap.parse_args()

    if OUT_JSON.exists() and not args.force:
        print(f"OSM-контекст уже есть: {OUT_JSON.relative_to(ROOT)} — пропускаю (перекачать: --force)")
        return 0

    lat0 = sum(v[0] for v in TURBINES.values()) / len(TURBINES)
    lon0 = sum(v[1] for v in TURBINES.values()) / len(TURBINES)
    query = QUERY.format(r=RADIUS_M, lat=f"{lat0:.6f}", lon=f"{lon0:.6f}")
    print(f"Overpass: радиус {RADIUS_M} м от {lat0:.6f}, {lon0:.6f} …")

    js, last_err = None, ""
    for attempt in range(3):
        try:
            r = requests.post(args.url, data={"data": query}, timeout=TIMEOUT_S,
                              headers={"User-Agent": "HackAlem-wind-viz/0.1 (hackathon demo)"})
            if r.status_code in (429, 502, 503, 504):     # перегрузка сервера — короткая пауза и повтор
                body = r.text.strip()
                last_err = f"HTTP {r.status_code}" + ("" if body.startswith("<") else f": {body[:200]}")  # HTML-страницы ошибок не печатаем
                print(f"  {last_err} — повтор через {10 * (attempt + 1)} с", file=sys.stderr)
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            js = r.json()
            break
        except (requests.RequestException, ValueError) as exc:
            last_err = str(exc)
            print(f"  ошибка запроса: {exc}", file=sys.stderr)
            time.sleep(5)
    if js is None:
        print(f"ОШИБКА: Overpass API недоступен ({args.url}): {last_err}\n"
              "Страница viz/index.html будет работать без слоёв OSM (турбины парка, лес, застройка).",
              file=sys.stderr)
        return 1

    elements = js.get("elements", [])
    kinds = {"turbines": 0, "forest": 0, "residential": 0}
    for el in elements:
        t = el.get("tags", {})
        if t.get("power") == "generator":
            kinds["turbines"] += 1
        elif t.get("landuse") == "residential":
            kinds["residential"] += 1
        else:
            kinds["forest"] += 1
    out = {
        "source": "OpenStreetMap contributors (ODbL), Overpass API",
        "url": args.url,
        "fetched_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "center": {"lat": round(lat0, 6), "lon": round(lon0, 6)},
        "radius_m": RADIUS_M,
        "query": query,
        "counts": kinds,
        "osm3s": js.get("osm3s", {}),
        "elements": elements,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False))
    print(f"Готово: {OUT_JSON.relative_to(ROOT)} — турбин {kinds['turbines']}, лесных полигонов {kinds['forest']}, "
          f"застройки {kinds['residential']} ({OUT_JSON.stat().st_size / 1024:.0f} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
