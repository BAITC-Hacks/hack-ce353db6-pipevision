"""Константы проекта: координаты, часовые пояса, горизонты, пути."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _detect_root() -> Path:
    """Корень проекта (где лежат data/, models/, outputs/).

    Порядок: переменная WIND_AGENT_ROOT → папка репозитория при установке `pip install -e .` →
    текущая папка (при обычной установке в site-packages пакет лежит вне репозитория).
    """
    env = os.environ.get("WIND_AGENT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for cand in (Path(__file__).resolve().parents[2], Path.cwd()):
        if (cand / "data" / "raw").is_dir():
            return cand
    return Path.cwd()


ROOT = _detect_root()
DATA_RAW = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache" / "openmeteo"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# Координаты турбин из ТЗ (maps.app.goo.gl/iN6svMt69D5qRpFU9, maps.app.goo.gl/8UQMwsYavY6nLvFY8)
TURBINES: dict[str, tuple[float, float]] = {
    "t1": (43.645150, 78.535604),
    "t2": (43.643198, 78.538828),
}
RAW_FILES = {"t1": DATA_RAW / "turbine_1.csv", "t2": DATA_RAW / "turbine_2.csv"}


@dataclass(frozen=True)
class Site:
    """Площадка ВЭС: координаты турбин и паспорт. Модель обучена на ВЭС «Нурлы» (has_history=True);
    для любой другой площадки прогноз строится переносом этой модели (обобщённая кривая мощности, точность ниже)."""
    key: str                                   # идентификатор для файлов и кэша (без пробелов)
    name: str
    turbines: dict = field(default_factory=dict)   # {ключ турбины: (широта, долгота)}
    rated_mw: float | None = None              # номинальная мощность одной турбины, МВт (для вывода в МВт·ч)
    n_turbines: int | None = None              # число турбин площадки (у «Нурлы» в данных две)
    has_history: bool = False                  # есть SCADA-история → модель обучена на этой площадке

    @property
    def series(self) -> list[str]:
        return list(self.turbines) + ["farm"]

    @property
    def transfer(self) -> bool:
        """Прогноз переносом модели (без истории площадки)."""
        return not self.has_history


NURLY = Site("nurly", "ВЭС «Нурлы» (Енбекшиказахский район)", TURBINES, rated_mw=2.5, n_turbines=2, has_history=True)
TRANSFER_TURBINE_ID = 0    # признак turbine_id для «чужой» площадки — кривая мощности турбины 1


def custom_site(lat: float, lon: float, n_turbines: int = 1, rated_mw: float | None = None, name: str | None = None) -> Site:
    """Произвольная площадка по координатам: одна виртуальная турбина `u1` (все турбины площадки попадают
    в одну ячейку сетки моделей погоды ~11 км), масштаб в МВт задают n_turbines × rated_mw."""
    lat, lon = float(lat), float(lon)
    if not (40.0 <= lat <= 56.0 and 46.0 <= lon <= 88.0):
        raise ValueError(f"координаты {lat:.4f}, {lon:.4f} вне территории Казахстана (широта 40–56, долгота 46–88)")
    key = f"c{lat:.3f}_{lon:.3f}".replace(".", "p").replace("-", "m")
    return Site(key, name or f"площадка {lat:.3f}, {lon:.3f}", {"u1": (lat, lon)}, rated_mw, int(n_turbines), False)

# Метки времени в датасете локальные. Казахстан перешёл на единое время UTC+5 1 марта 2024,
# до этого Алматинская область жила по UTC+6. Подтверждено кросс-корреляцией с Open-Meteo (docs/01_analysis.md).
TZ_SWITCH_LOCAL = "2024-03-01 00:00:00"
UTC_OFFSET_BEFORE = 6
UTC_OFFSET_AFTER = 5
LOCAL_TZ_NAME = "Asia/Almaty (UTC+5)"

# Горизонт прогноза и протокол теста из ТЗ
HORIZON_HOURS = 48
ISSUE_HOUR_LOCAL = 23          # прогноз «на день D» выпускается в конце дня D (23:00 местного) на D+1 и D+2
TEST_ISSUE_START = "2026-01-31"
TEST_ISSUE_END = "2026-02-27"   # выпуск 27.02 закрывает 28.02 и 01.03; факт есть до 31.01
HISTORY_END = "2026-01-31"     # последний день фактических данных

# Погодные переменные Open-Meteo (единицы: м/с, °C, гПа, градусы)
WEATHER_VARS = [
    "wind_speed_10m", "wind_speed_80m", "wind_speed_100m", "wind_speed_120m",
    "wind_direction_100m", "wind_gusts_10m", "temperature_2m", "surface_pressure",
]
LEAD_DAYS = (1, 2)             # previous_day1 → часы 1–24, previous_day2 → часы 25–48

# Ансамбль моделей погоды: помимо best_match (лучшая модель по мнению Open-Meteo) берём три глобальные
# модели по отдельности. Их среднее коррелирует с замером ветра на 0.78 против 0.71 у одной модели,
# MAE мощности ниже на ~5 % (см. docs/01_analysis.md). Если какой-то модели нет, признаки заполняются best_match.
ENSEMBLE_MODELS = ["gfs_seamless", "icon_seamless", "ecmwf_ifs025"]
ENSEMBLE_VARS = ["wind_speed_100m", "wind_speed_10m", "temperature_2m"]
PREVIOUS_RUNS_START = "2024-02-16"  # с этой даты Open-Meteo хранит архив предыдущих запусков

# Фильтр качества обучающей выборки: простои/ограничения при сильном ветре
CURTAIL_WIND_MS = 8.0
CURTAIL_POWER = 0.02
MIN_SAMPLES_PER_HOUR = 4

# Бэктест с эталоном (факт есть только до 31.01.2026)
BACKTEST_TRAIN_END = "2025-11-30"
BACKTEST_CALIB_START = "2025-10-01"   # последние 2 месяца обучения — калибровка интервалов (конформная поправка)
BACKTEST_TEST_START = "2025-12-01"
BACKTEST_TEST_END = "2026-01-31"
