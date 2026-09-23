"""Сборка атласа ветрового ресурса Казахстана (NASA POWER, MERRA-2 2001–2020) → data/atlas/kz_wind_atlas.csv.

Запуск: python scripts/build_wind_atlas.py   (≈10 запросов к NASA POWER, ключ не нужен)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_agent import atlas  # noqa: E402
from wind_agent.config import TURBINES  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    df = atlas.build_atlas(force=True, progress=lambda i, n, t: print(f"[{i}/{n}] {t}", flush=True))
    kz = df[df["inside_kz"]]
    print(f"точек всего: {len(df)}, в маске Казахстана: {len(kz)}")
    lat, lon = TURBINES["t1"]
    print("Нурлы:", "внутри" if atlas.in_kazakhstan(lat, lon) else "СНАРУЖИ",
          json.dumps(atlas.nearest_cell(lat, lon), ensure_ascii=False))
    print(kz["resource_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
