"""Мост между панелью оператора и ядром wind_agent: один вызов run_forecast(params) → ForecastResult.

UI не зависит от версии API ядра и от имён рядов прогноза:
* параметры site / issue_hour / out_dir передаются в orchestrator.run_cycle, только если он их принимает
  (проверка через inspect.signature при каждом запуске); иначе прогноз строится в текущем режиме ядра,
  а в ForecastResult.notices попадает предупреждение;
* ряды берутся из данных: все значения колонки `turbine`, кроме `farm`, — турбины;
* файлы прогона (csv и md) пишутся в outputs/tmp/ui, а не в outputs/forecasts и outputs/reports:
  если run_cycle не принимает out_dir, на время запуска каталог записи подменяется в этом процессе
  (код ядра не меняется).

Модуль не импортирует streamlit — его можно вызывать из тестов и скриптов.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import json
import logging
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
try:
    import wind_agent  # noqa: F401
except ImportError:                             # запуск без `pip install -e .`
    sys.path.insert(0, str(_REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

from wind_agent import config  # noqa: E402
from wind_agent.agent import llm, orchestrator, tools  # noqa: E402
from wind_agent.weather import WeatherClient  # noqa: E402

load_dotenv(config.ROOT / ".env")               # как cli.py: OPENAI_API_KEY / OPENAI_MODEL из .env

NURLY_KEY, CUSTOM_KEY = "nurly", "custom"
MODE_REPLAY, MODE_LIVE = "replay", "live"
FARM = "farm"
REPLAY_MIN, REPLAY_MAX, REPLAY_DEFAULT = date(2024, 3, 1), date(2026, 2, 27), date(2026, 2, 10)
KZ_LAT, KZ_LON = (40.0, 56.0), (46.0, 88.0)     # границы, которые принимает config.custom_site
NURLY_FALLBACK_RATED_MW = 2.5                   # T1/T2 — Goldwind GW109/2500 (README, раздел 9)
UI_OUT_DIR = Path(os.environ.get("WIND_AGENT_UI_OUT") or config.OUTPUTS_DIR / "tmp" / "ui")

TRANSFER_WARNING = ("Перенос модели ВЭС «Нурлы»: истории этой площадки нет, кривая мощности обобщённая, ошибка выше; "
                    "для точного прогноза загрузите SCADA")
DECISION_RU = {"accept": "принять прогноз", "flag": "принять с флагом для диспетчера",
               "recalculate": "пересчитать (повторный запрос входа)"}

_RUN_LOCK = threading.Lock()                    # один прогон за раз: подмена каталога записи — глобальная для процесса


# ---------------------------------------------------------------- параметры и результат
@dataclass(frozen=True)
class ForecastParams:
    site: str = NURLY_KEY                       # nurly | custom
    lat: float = 48.0
    lon: float = 68.0
    n_turbines: int = 1
    rated_mw: float = 2.5
    name: str = ""
    mode: str = MODE_REPLAY                     # replay | live
    issue_date: date = REPLAY_DEFAULT
    issue_hour: int = config.ISSUE_HOUR_LOCAL
    horizon: int = 48                           # 24 | 48 — фильтр lead_hours на стороне UI
    use_llm: bool = False

    def run_key(self) -> tuple:
        """Параметры, от которых зависит прогон ядра (горизонт — только фильтр отображения)."""
        site = (self.site,) if self.site == NURLY_KEY else (self.site, round(self.lat, 4), round(self.lon, 4),
                                                           self.n_turbines, round(self.rated_mw, 2), self.name.strip())
        when = (self.mode,) if self.mode == MODE_LIVE else (self.mode, self.issue_date, self.issue_hour)
        return site + when + (self.use_llm,)


@dataclass(frozen=True)
class SiteInfo:
    key: str
    name: str
    points: list                                # [(ключ турбины, широта, долгота)]
    rated_mw: float | None                      # номинал одной турбины, МВт
    n_turbines: int                             # турбин на площадке (масштаб ряда farm)
    transfer: bool                              # прогноз переносом модели (истории площадки нет)
    custom_requested: bool = False              # оператор выбрал свою площадку
    custom_applied: bool = False                # ядро построило прогноз именно для неё

    @property
    def short_name(self) -> str:
        return self.name.split(" (")[0]

    def capacity_mw(self, series: str) -> float | None:
        """Номинал ряда в МВт: турбина — rated_mw, парк — rated_mw × число турбин."""
        if not self.rated_mw:
            return None
        return self.rated_mw * (self.n_turbines if series == FARM else 1)


@dataclass
class ForecastResult:
    params: ForecastParams
    site: SiteInfo
    forecast: pd.DataFrame                      # прогноз ядра: контракт FORECAST_COLUMNS, все ряды, полный горизонт
    analysis: dict                              # tools.analyze_forecast (по полному горизонту)
    final: dict                                 # decision, reasoning, narrative, report, csv, llm_used, rules_decision
    trace: list                                 # ход работы агента
    meta: dict                                  # метаданные погоды: источник, время выпуска, хэш входа
    report_md: str
    notices: list = field(default_factory=list)       # чего ядро пока не умеет / что пошло не так
    core_log: list = field(default_factory=list)      # предупреждения ядра за время прогона: [(уровень, текст)]
    previous_issue: str | None = None
    previous: pd.DataFrame | None = None        # предыдущий выпуск (для ревизии и пунктира на графике)
    ens_std: pd.Series | None = None            # разброс ансамбля GFS/ICON/ECMWF, ветер 100 м (м/с), индекс — UTC
    api_calls: int = 0
    elapsed_s: float = 0.0

    @property
    def turbines(self) -> list[str]:
        return [s for s in pd.unique(self.forecast["turbine"]) if s != FARM]

    @property
    def series(self) -> list[str]:
        return ([FARM] if FARM in set(self.forecast["turbine"]) else []) + self.turbines

    @property
    def issue_label(self) -> str:
        return str(self.meta.get("issue_date") or self.forecast["issue_date"].iloc[0])

    def view(self, horizon: int | None = None) -> pd.DataFrame:
        """Прогноз для отображения: lead_hours ≤ горизонта + колонка `time_local` (наивное местное время)."""
        h = int(horizon or self.params.horizon)
        v = self.forecast[self.forecast["lead_hours"] <= h].copy()
        v["time_local"] = pd.to_datetime(v["target_time_local"].astype(str).str[:19])
        return v.sort_values(["turbine", "lead_hours"]).reset_index(drop=True)


# ---------------------------------------------------------------- возможности ядра
def _kw_names(fn) -> set[str]:
    try:
        ps = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return set()
    return {p.name for p in ps if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def capabilities() -> dict:
    """Что умеет текущая версия ядра (смотрим сигнатуру run_cycle, а не номер версии)."""
    names = _kw_names(orchestrator.run_cycle)
    return {"site": "site" in names and hasattr(config, "custom_site"),
            "issue_hour": "issue_hour" in names,
            "out_dir": "out_dir" in names}


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def llm_model_name() -> str:
    return os.getenv("OPENAI_MODEL") or getattr(llm, "DEFAULT_MODEL", "")


def load_core() -> dict:
    """Прогреть модель и климатологию SCADA (оба кэшируются в процессе; в UI — ещё и st.cache_resource)."""
    model = tools.load_model()
    try:
        tools.load_history()
    except Exception:  # noqa: BLE001 — климатология нужна анализу, но прогрев необязателен
        pass
    return {"model": model, "model_version": tools.model_version(model)}


def nurly_site() -> "SiteInfo":
    """Паспорт ВЭС «Нурлы» (для вкладок, которые показывают результаты replay независимо от выбранной площадки)."""
    return _site_info(getattr(config, "NURLY", None), ForecastParams())


def nurly_points() -> list:
    site = getattr(config, "NURLY", None)
    turbines = getattr(site, "turbines", None) or config.TURBINES
    return [(k, float(lat), float(lon)) for k, (lat, lon) in turbines.items()]


# ---------------------------------------------------------------- площадка
def _site_info(site_obj, params: ForecastParams) -> SiteInfo:
    custom = params.site == CUSTOM_KEY
    if site_obj is None:                        # старое ядро: только «Нурлы»
        pts = nurly_points()
        return SiteInfo(NURLY_KEY, "ВЭС «Нурлы»", pts, NURLY_FALLBACK_RATED_MW, len(pts), False, custom, False)
    turbines = getattr(site_obj, "turbines", {}) or {}
    pts = [(k, float(lat), float(lon)) for k, (lat, lon) in turbines.items()]
    rated = getattr(site_obj, "rated_mw", None)
    n = getattr(site_obj, "n_turbines", None) or len(pts) or 1
    transfer = bool(getattr(site_obj, "transfer", not getattr(site_obj, "has_history", False)))
    return SiteInfo(str(getattr(site_obj, "key", NURLY_KEY)), str(getattr(site_obj, "name", "площадка")), pts,
                    float(rated) if rated else None, int(n), transfer, custom, custom)


def _previous_forecast(params: ForecastParams, site: SiteInfo) -> tuple[pd.DataFrame | None, str | None]:
    """Предыдущий выпуск для анализа ревизии (только чтение, как в run_replay/run_live). Только для «Нурлы»:
    для своей площадки прошлых выпусков нет."""
    if site.transfer or site.custom_applied:
        return None, None
    try:
        if params.mode == MODE_LIVE:
            files = sorted(p for p in (config.OUTPUTS_DIR / "live").glob("*Z.csv"))
            path = files[-1] if files else None
        else:
            prev = (params.issue_date - timedelta(days=1)).isoformat()
            path = config.OUTPUTS_DIR / "forecasts" / f"{prev}.csv"
        if path is None or not path.exists():
            return None, None
        df = pd.read_csv(path)
        expected = {k for k, _, _ in site.points} | {FARM}
        if df.empty or not set(df["turbine"]) <= expected:
            return None, None
        return df, str(df["issue_date"].iloc[0])
    except Exception:  # noqa: BLE001 — без ревизии прогноз всё равно строится
        return None, None


# ---------------------------------------------------------------- запись и журнал на время прогона
@contextlib.contextmanager
def _redirect_outputs(out_dir: Path):
    """В replay ядро пишет в outputs/forecasts и outputs/reports без параметра каталога. На время прогона
    подменяем out_dir у tools.save_forecast / tools.write_report (только в процессе UI)."""
    names = ("save_forecast", "write_report")
    orig = {n: getattr(tools, n) for n in names}

    def wrap(fn, sub):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            ba = sig.bind_partial(*args, **kwargs)
            ba.arguments["out_dir"] = out_dir / sub
            return fn(*ba.args, **ba.kwargs)
        return inner

    tools.save_forecast = wrap(orig["save_forecast"], "forecasts")
    tools.write_report = wrap(orig["write_report"], "reports")
    try:
        yield
    finally:
        for n, fn in orig.items():
            setattr(tools, n, fn)


class _CoreLogHandler(logging.Handler):
    """Шаги агента (DEBUG-строка `_record` оркестратора) → progress; WARNING и выше → в результат."""

    def __init__(self, progress, thread):
        super().__init__(logging.DEBUG)
        self.progress, self.thread, self.records = progress, thread, []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage().strip()
        except Exception:  # noqa: BLE001
            return
        if record.levelno >= logging.WARNING:
            self.records.append((record.levelname.lower(), msg))
        if (self.progress and record.levelno == logging.DEBUG and record.name.endswith("orchestrator")
                and threading.current_thread() is self.thread):
            try:
                self.progress(msg)
            except Exception:  # noqa: BLE001 — индикатор хода не должен ронять прогон
                pass


@contextlib.contextmanager
def _capture_core_log(progress):
    logger = logging.getLogger("wind_agent")
    handler = _CoreLogHandler(progress, threading.current_thread())
    old_level, old_prop = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False                    # DEBUG ядра не уходит в корневой логгер сервера
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_prop


# ---------------------------------------------------------------- прогон
def run_forecast(params: ForecastParams, wx: WeatherClient | None = None, progress=None) -> ForecastResult:
    """Один выпуск прогноза через orchestrator.run_cycle с параметрами панели оператора.

    progress(msg) вызывается на каждом шаге агента (из того же потока). Исключения ядра пробрасываются.
    """
    t0 = time.perf_counter()
    caps = capabilities()
    notices: list[str] = []
    wx = wx or WeatherClient(offline=False)
    live = params.mode == MODE_LIVE
    kwargs: dict = {}

    site_obj = None
    if params.site == CUSTOM_KEY:
        if caps["site"]:
            site_obj = config.custom_site(params.lat, params.lon, n_turbines=int(params.n_turbines),
                                          rated_mw=float(params.rated_mw), name=params.name.strip() or None)
            kwargs["site"] = site_obj
        else:
            notices.append("Своя площадка пока недоступна: текущая версия ядра не принимает параметр site — "
                           "прогноз построен для ВЭС «Нурлы».")
    elif caps["site"] and getattr(config, "NURLY", None) is not None:
        site_obj = config.NURLY
        kwargs["site"] = site_obj
    site = _site_info(site_obj, params)

    if live:
        kwargs.update(live=True, out_name=f"ui_live_{site.key}")
    else:
        kwargs["issue_date"] = params.issue_date.isoformat()
        if caps["issue_hour"]:
            kwargs["issue_hour"] = int(params.issue_hour)
        elif int(params.issue_hour) != config.ISSUE_HOUR_LOCAL:
            notices.append(f"Час выпуска пока фиксирован: текущая версия ядра не принимает issue_hour — прогноз выпущен "
                           f"в {config.ISSUE_HOUR_LOCAL}:00 местного времени, а не в {int(params.issue_hour):02d}:00.")

    previous, previous_issue = _previous_forecast(params, site)
    if previous is not None:
        kwargs["previous"] = previous

    if params.use_llm and llm_available():
        settings = llm.llm_settings()
        llm_state = orchestrator.LLMState(settings=settings) if settings else orchestrator.LLMState()
    else:
        llm_state = orchestrator.LLMState()
        if params.use_llm:
            notices.append("OPENAI_API_KEY не найден — агент отработал по детерминированным правилам.")

    out_dir = UI_OUT_DIR / ("live" if live else "replay") / site.key
    redirect = contextlib.nullcontext()
    if caps["out_dir"]:
        kwargs["out_dir"] = out_dir
    else:
        redirect = _redirect_outputs(out_dir)

    with _RUN_LOCK, _capture_core_log(progress) as core_log, redirect:
        calls0 = wx.calls
        session = orchestrator.run_cycle(wx, llm_state, **kwargs)
        api_calls = int(wx.calls - calls0)
        report_path = Path(session.final["report"])
        report_md = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    ens_std = None
    try:                                        # признак ens_std_ws100 из prepare_data (среднее по турбинам)
        feats = (session.state.get("prepared") or {}).get("features") or {}
        cols = [X["ens_std_ws100"] for X in feats.values() if "ens_std_ws100" in getattr(X, "columns", [])]
        if cols:
            ens_std = pd.concat(cols, axis=1).mean(axis=1)
            ens_std.index = pd.to_datetime(ens_std.index, utc=True)
    except Exception:  # noqa: BLE001 — разброс ансамбля необязателен
        ens_std = None

    if params.use_llm and any(t.get("tool") == "llm" and t.get("status") == "error" for t in session.trace):
        notices.append("LLM недоступна или вернула ошибку — оставшиеся шаги агент выполнил по детерминированным правилам.")

    return ForecastResult(
        params=params, site=site, forecast=session.state["forecast"].copy(), analysis=session.state["analysis"],
        final=dict(session.final), trace=list(session.trace), meta=dict(session.state.get("weather", {}).get("meta", {})),
        report_md=report_md, notices=notices, core_log=list(core_log), previous_issue=previous_issue, previous=previous,
        ens_std=ens_std,
        api_calls=api_calls, elapsed_s=round(time.perf_counter() - t0, 2))


# ---------------------------------------------------------------- агрегаты для отображения
def _sum(x: pd.Series) -> float:
    """Сумма с округлением до 0.01 — как в analyze_forecast, чтобы числа панели совпадали с отчётом агента."""
    return round(float(x.sum()), 2)


def series_stats(view: pd.DataFrame, series: str) -> dict:
    """Энергия (часы номинала) и мощность ряда по отфильтрованному горизонту; сутки — по упреждению 1–24 / 25–48 ч."""
    d = view[view["turbine"] == series]
    day1, day2 = d[d["lead_hours"] <= 24], d[d["lead_hours"] > 24]
    return {"hours": int(len(d)), "energy": _sum(d["p50"]), "energy_p10": _sum(d["p10"]),
            "energy_p90": _sum(d["p90"]), "day1": _sum(day1["p50"]), "day1_hours": int(len(day1)),
            "day2": _sum(day2["p50"]) if len(day2) else None, "day2_hours": int(len(day2)),
            "mean_p50": round(float(d["p50"].mean()), 3) if len(d) else float("nan"),
            "max_p50": round(float(d["p50"].max()), 3) if len(d) else float("nan"),
            "band": round(float((d["p90"] - d["p10"]).mean()), 3) if len(d) else float("nan"),
            "wind_mean": round(float(d["wind_speed_100m"].mean()), 1) if len(d) else float("nan"),
            "wind_max": round(float(d["wind_speed_100m"].max()), 1) if len(d) else float("nan")}


def daily_table(view: pd.DataFrame, series: list[str], label, scale=None, unit: str = "ч.н.") -> pd.DataFrame:
    """Энергия по местным суткам: «P50 [P10–P90]» для каждого ряда + строка «Итого». scale(series) → множитель."""
    v = view.assign(day=view["time_local"].dt.strftime("%Y-%m-%d"))
    days = sorted(v["day"].unique())
    hours = v[v["turbine"] == series[0]].groupby("day").size()
    rows = []
    for day in days + [None]:
        part = v if day is None else v[v["day"] == day]
        n = int(hours.sum() if day is None else hours.get(day, 0))
        row = {"Сутки (местные)": f"Итого {n} ч" if day is None else (f"{day}" + ("" if n == 24 else f" ({n} ч)"))}
        for s in series:
            k = (scale(s) if scale else 1.0) or 1.0
            d = part[part["turbine"] == s]
            row[f"{label(s)}, {unit}"] = f"{_sum(d['p50']) * k:.1f} [{_sum(d['p10']) * k:.1f}–{_sum(d['p90']) * k:.1f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def hourly_table(view: pd.DataFrame, series: list[str], label, scale=None, unit: str = "доля ном.") -> pd.DataFrame:
    """Почасовая таблица: время, упреждение, P10/P50/P90 по рядам (× scale(series) — в МВт), погода по первому ряду."""
    base = view[view["turbine"] == series[0]].sort_values("lead_hours")
    out = pd.DataFrame({"Местное время": base["time_local"].values,
                        "Упреждение, ч": base["lead_hours"].astype(int).values})
    for s in series:
        d = view[view["turbine"] == s].sort_values("lead_hours")
        k = (scale(s) if scale else 1.0) or 1.0
        for q in ("p10", "p50", "p90"):
            out[f"{label(s)} {q.upper()}, {unit}"] = (d[q] * k).round(3).values
    for col, name in (("wind_speed_100m", "Ветер 100 м, м/с"), ("wind_direction_100m", "Направление 100 м, °"),
                      ("wind_speed_10m", "Ветер 10 м, м/с"), ("wind_gusts_10m", "Порывы 10 м, м/с"),
                      ("temperature_2m", "Температура, °C"), ("surface_pressure", "Давление, гПа")):
        if col in base:
            out[name] = base[col].values
    return out


def forecast_csv(view: pd.DataFrame) -> bytes:
    """CSV в контракте FORECAST_COLUMNS (порядок колонок как в outputs/forecasts/*.csv)."""
    cols = [c for c in tools.FORECAST_COLUMNS if c in view.columns]
    return view[cols].to_csv(index=False).encode("utf-8")


def trace_table(trace: list) -> pd.DataFrame:
    return pd.DataFrame([{"Шаг": t.get("step", i + 1), "Инструмент": t.get("tool", ""),
                          "Кто вызвал": "LLM" if t.get("llm_used") else "правила", "Статус": t.get("status", ""),
                          "Время, с": float(t.get("duration_s") or 0.0), "Итог": str(t.get("summary", ""))}
                         for i, t in enumerate(trace)])


def _dm(local: str | None) -> str:
    """«2026-02-11 06:00» → «11.02 06:00»."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}:\d{2})", str(local or ""))
    return f"{m.group(3)}.{m.group(2)} {m.group(4)}" if m else str(local or "")


def flag_text(flag: dict, analysis: dict) -> str:
    """Короткая формулировка флага для ленты предупреждений (числа — из анализа)."""
    code, msg = flag.get("code", ""), str(flag.get("message", ""))
    farm = (analysis.get("metrics") or {}).get(FARM, {})
    rev = analysis.get("revision") or {}
    try:
        if code == "ramp":
            return f"Рампа P50 {farm['max_ramp']:.2f} доли ном./ч · {_dm(farm.get('max_ramp_time_local'))}"
        if code == "revision" and FARM in rev:
            r = rev[FARM]
            return f"Ревизия к выпуску {rev['previous_issue_date']}: MAE {r['mae']:.2f}, смещение {r['bias']:+.2f}"
        if code == "low_confidence":
            return f"Широкий интервал P10–P90: {farm['mean_band_p90_p10']:.2f} доли ном."
        if code == "climatology":
            return f"Отклонение от нормы месяца {farm['vs_climatology_pct']:+.0f} %"
        if code == "calm_window":
            m = re.search(r"(\d+)\s*ч.*?с\s+(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2})", msg)
            if m:
                return f"Штиль {m.group(1)} ч с {_dm(m.group(2))}"
        if code == "ensemble_gap":
            return "Неполный ансамбль GFS/ICON/ECMWF"
        if code == "transfer":
            return "Перенос модели на площадку без истории"
    except (KeyError, TypeError, ValueError):
        pass
    return msg if len(msg) <= 120 else msg[:117] + "…"


# ---------------------------------------------------------------- пороги агента, окна и алерты
RAMP_THRESHOLD = tools.RAMP_THRESHOLD           # |ΔP50| за час, доля номинала — те же пороги, что у analyze_forecast
CALM_LEVEL = tools.CALM_LEVEL
CALM_MIN_HOURS = tools.CALM_MIN_HOURS
WIDE_BAND_THRESHOLD = tools.WIDE_BAND_THRESHOLD
REVISION_MAE_THRESHOLD = tools.REVISION_MAE_THRESHOLD
HIGH_CONFIDENCE_BAND = WIDE_BAND_THRESHOLD / 2  # P90−P10 ниже — «высокая» уверенность


def confidence_word(band: float) -> str:
    if band > WIDE_BAND_THRESHOLD:
        return "низкая"
    return "высокая" if band < HIGH_CONFIDENCE_BAND else "средняя"


def ramp_windows(d: pd.DataFrame) -> list[dict]:
    """Часы с |ΔP50| > RAMP_THRESHOLD (d — один ряд, по возрастанию lead_hours, колонка time_local)."""
    d = d.sort_values("lead_hours")
    p, t = d["p50"].to_numpy(), list(d["time_local"])
    return [{"t0": t[i - 1], "t1": t[i], "value": float(abs(p[i] - p[i - 1])), "sign": float(p[i] - p[i - 1])}
            for i in range(1, len(p)) if abs(p[i] - p[i - 1]) > RAMP_THRESHOLD]


def calm_windows(d: pd.DataFrame) -> list[dict]:
    """Серии ≥ CALM_MIN_HOURS часов подряд с P50 ≤ CALM_LEVEL."""
    d = d.sort_values("lead_hours")
    p, t = d["p50"].to_numpy(), list(d["time_local"])
    out, start = [], None
    for i, v in enumerate(list(p) + [np.inf]):
        if v <= CALM_LEVEL and start is None:
            start = i
        elif v > CALM_LEVEL and start is not None:
            if i - start >= CALM_MIN_HOURS:
                out.append({"t0": t[start], "t1": t[i - 1], "hours": i - start})
            start = None
    return out


def _span(t0, t1) -> str:
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    return f"{t0:%d.%m %H:%M}–{t1:%H:%M}" if t0.date() == t1.date() else f"{t0:%d.%m %H:%M}–{t1:%d.%m %H:%M}"


_ALERT_ERROR = {"range_violation", "quantile_order", "incomplete"}


def alert_rows(res: "ForecastResult", view: pd.DataFrame, custom_banner: bool = False) -> list[dict]:
    """Лента предупреждений: уровень, заголовок, интервал (местное время), значение и порог, действие.
    Флаги — из анализа агента (ряд farm), интервалы рамп и штиля — по отображаемому горизонту."""
    a = res.analysis
    farm_v = view[view["turbine"] == FARM] if FARM in set(view["turbine"]) else view[view["turbine"] == res.series[0]]
    t_first, t_last = farm_v["time_local"].min(), farm_v["time_local"].max()
    cap = res.site.capacity_mw(FARM)
    rows = []
    for n in res.notices:
        rows.append({"level": "warning", "title": "Ограничение ядра", "interval": "", "value": n, "action": "", "t": t_first})
    for lvl, msg in res.core_log:
        if lvl in ("error", "critical"):
            rows.append({"level": "error", "title": "Ошибка ядра", "interval": "", "value": msg[:140], "action": "", "t": t_first})
    for f in a.get("flags", []):
        code, lvl, msg = f.get("code", ""), f.get("level", "info"), str(f.get("message", ""))
        level = "error" if code in _ALERT_ERROR else ("warning" if lvl == "warning" else "info")
        row = {"level": level, "code": code, "title": code, "interval": _span(t_first, t_last), "value": msg[:140],
               "action": "", "t": t_first}
        if code == "ramp":
            ramps = ramp_windows(farm_v) or []
            top = max(ramps, key=lambda r: r["value"]) if ramps else None
            mw = f" ({top['value'] * cap:.1f} МВт/ч)" if top and cap else ""
            row.update(title="Резкая рампа",
                       interval=_span(top["t0"], top["t1"]) if top else row["interval"],
                       value=(f"|ΔP50| {top['value']:.2f} доли ном./ч{mw}, порог {RAMP_THRESHOLD:.2f}; всего {len(ramps)}"
                              if top else msg[:140]),
                       action=f"проверить график выдачи {_span(top['t0'], top['t1'])}" if top else "проверить график выдачи",
                       t=top["t0"] if top else t_first)
        elif code == "revision" and (a.get("revision") or {}).get(FARM):
            r = a["revision"][FARM]
            common = farm_v[farm_v["lead_hours"] <= r.get("n_hours", 24)]
            row.update(title="Существенная ревизия",
                       interval=_span(common["time_local"].min(), common["time_local"].max()) if len(common) else row["interval"],
                       value=f"MAE {r['mae']:.2f} к выпуску {a['revision']['previous_issue_date']}, смещение {r['bias']:+.2f}; "
                             f"порог {REVISION_MAE_THRESHOLD:.2f}",
                       action="сверить план выдачи на общие часы с выпуском D−1")
        elif code == "calm_window":
            calms = calm_windows(farm_v)
            top = max(calms, key=lambda c: c["hours"]) if calms else None
            row.update(title="Окно штиля для ТО",
                       interval=_span(top["t0"], top["t1"]) if top else row["interval"],
                       value=(f"{top['hours']} ч с P50 ≤ {CALM_LEVEL:.2f}; минимум {CALM_MIN_HOURS} ч" if top else msg[:140]),
                       action="рассмотреть окно для ТО", t=top["t0"] if top else t_first)
        elif code == "low_confidence":
            m = (a.get("metrics") or {}).get(FARM, {})
            row.update(title="Низкая уверенность",
                       value=f"P90−P10 {m.get('mean_band_p90_p10', float('nan')):.2f} доли ном., порог {WIDE_BAND_THRESHOLD:.2f}",
                       action="учесть резерв мощности")
        elif code == "climatology":
            m = (a.get("metrics") or {}).get(FARM, {})
            row.update(title="Отклонение от нормы месяца",
                       value=f"{m.get('vs_climatology_pct', 0):+.0f} % к норме {m.get('climatology_mean', float('nan')):.2f}, порог ±50 %")
        elif code == "input_quality":
            row.update(title="Качество входных данных", action="повторный запрос погоды")
        elif code in _ALERT_ERROR:
            row.update(title={"range_violation": "Значения вне [0, 1]", "quantile_order": "Нарушен порядок квантилей",
                              "incomplete": "Неполный прогноз"}[code], action="пересчитать")
        elif code == "ensemble_gap":
            row.update(title="Неполный ансамбль погоды")
        elif code == "transfer":
            if custom_banner:
                continue
            row.update(title="Перенос модели")
        rows.append(row)
    order = {"error": 0, "warning": 1, "info": 2}
    return sorted(rows, key=lambda r: (order.get(r["level"], 3), pd.Timestamp(r["t"])))


# ---------------------------------------------------------------- факт SCADA и точность
@functools.lru_cache(maxsize=1)
def nurly_actual() -> dict:
    """Часовой факт SCADA «Нурлы» (доля номинала, индекс UTC): t1, t2 и farm = среднее. До 31.01.2026."""
    from wind_agent.evaluate import load_actual_hourly
    act = {t: load_actual_hourly(p) for t, p in config.RAW_FILES.items() if Path(p).exists()}
    if len(act) > 1:
        act[FARM] = pd.concat(act.values(), axis=1).mean(axis=1)
    return act


def actual_for(view: pd.DataFrame, series: str, actual: dict) -> pd.Series | None:
    """Факт на часы прогноза ряда (по target_time_utc) или None, если факта нет."""
    s = actual.get(series)
    if s is None:
        return None
    d = view[view["turbine"] == series]
    y = s.reindex(pd.to_datetime(d["target_time_utc"], utc=True)).to_numpy()
    return None if np.isnan(y).all() else pd.Series(y, index=d.index)


def accuracy(forecast: pd.DataFrame, actual: dict) -> dict:
    """Метрики прогноза против факта (wind_agent.evaluate.evaluate): MAE, RMSE, смещение, покрытие P10–P90."""
    from wind_agent.evaluate import evaluate
    fc = forecast.copy()
    fc["t_utc"] = pd.to_datetime(fc["target_time_utc"], utc=True)
    act = {k: v for k, v in actual.items() if k != FARM}
    return evaluate(fc, act)


def load_uploaded_actual(files) -> dict:
    """Загруженные файлы факта в формате data/raw/turbine_*.csv → {t1|t2: часовой ряд}. Турбина — по имени файла."""
    from wind_agent.evaluate import load_actual_hourly
    out = {}
    for i, f in enumerate(files):
        name = getattr(f, "name", f"file{i}")
        key = "t2" if "2" in Path(name).stem else "t1"
        if key in out:
            key = f"t{len(out) + 1}"
        out[key] = load_actual_hourly(f)
    return out


# ---------------------------------------------------------------- «почему такой прогноз»
WORK_ZONE_MS = (6.0, 12.0)                      # рабочая зона кривой мощности: от заметной выработки до номинала
ENS_STD_TERCILES = (2.1, 2.4)                   # терцили среднего ens_std_ws100 по 28 выпускам 31.01–27.02.2026, м/с
CONE_HALF_DEG, CONE_RANGE_M, OSM_MATCH_M = 15.0, 2000.0, 80.0   # как в viz/index.html и scripts/build_viz_data.py
OSM_CONTEXT = config.ROOT / "data" / "terrain" / "osm_context.json"
COMPASS_RU = ["С", "ССВ", "СВ", "ВСВ", "В", "ВЮВ", "ЮВ", "ЮЮВ", "Ю", "ЮЮЗ", "ЮЗ", "ЗЮЗ", "З", "ЗСЗ", "СЗ", "ССЗ"]
MONTHS_RU = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь",
             "декабрь"]


def compass(deg: float) -> str:
    return COMPASS_RU[int(((deg % 360) + 11.25) // 22.5) % 16]


@functools.lru_cache(maxsize=1)
def park_layout() -> dict | None:
    """T1/T2 и остальные турбины парка из OSM в локальных метрах (x — восток, y — север) от центра osm_context."""
    if not OSM_CONTEXT.exists():
        return None
    d = json.loads(OSM_CONTEXT.read_text(encoding="utf-8"))
    lat0, lon0 = float(d["center"]["lat"]), float(d["center"]["lon"])
    kx, ky = 111320 * math.cos(math.radians(lat0)), 111132
    ours = {k: ((lon - lon0) * kx, (lat - lat0) * ky) for k, lat, lon in nurly_points()}
    others = []
    for el in d.get("elements", []):
        if el.get("type") == "node" and (el.get("tags") or {}).get("power") == "generator":
            x, y = (float(el["lon"]) - lon0) * kx, (float(el["lat"]) - lat0) * ky
            if min(math.hypot(x - ox, y - oy) for ox, oy in ours.values()) > OSM_MATCH_M:
                others.append((x, y))
    return {"ours": ours, "others": others}


def upwind_count(wd: float, layout: dict) -> int:
    """Турбины (без повторов) в конусе ±15° до 2 км навстречу ветру от T1/T2 — как upwindHits в viz/index.html."""
    targets = [("t:" + k, x, y) for k, (x, y) in layout["ours"].items()] + \
              [(f"osm{i}", x, y) for i, (x, y) in enumerate(layout["others"])]
    hit = set()
    for k, (ox, oy) in layout["ours"].items():
        for tid, x, y in targets:
            if tid == "t:" + k:
                continue
            dx, dy = x - ox, y - oy
            dist = math.hypot(dx, dy)
            if dist < 1 or dist > CONE_RANGE_M:
                continue
            brg = (math.degrees(math.atan2(dx, dy)) + 360) % 360
            if abs(((brg - wd + 540) % 360) - 180) <= CONE_HALF_DEG:
                hit.add(tid)
    return len(hit)


def why_lines(res: "ForecastResult", view: pd.DataFrame) -> list[tuple[str, str, str]]:
    """4–6 строк «показатель — значение — что это значит для выработки» из данных (без LLM)."""
    main = FARM if FARM in set(view["turbine"]) else res.series[0]
    d = view[view["turbine"] == main].sort_values("lead_hours")
    out = []
    ws = d["wind_speed_100m"].astype(float)
    if ws.notna().any():
        share = float(((ws >= WORK_ZONE_MS[0]) & (ws <= WORK_ZONE_MS[1])).mean())
        low, high = float((ws < WORK_ZONE_MS[0]).mean()), float((ws > WORK_ZONE_MS[1]).mean())
        if share >= 0.5:
            mean_txt = "основная выработка на рабочем участке кривой, чувствительна к ошибке ветра"
        elif low >= 0.5:
            mean_txt = "преобладает слабый ветер — выработка ниже половины номинала"
        elif high >= 0.5:
            mean_txt = "преобладает сильный ветер — работа около номинала"
        else:
            mean_txt = "режим переменный — выработка неравномерна по часам"
        out.append(("Ветер 100 м", f"{ws.mean():.1f} м/с, {100 * share:.0f} % часов в зоне "
                                   f"{WORK_ZONE_MS[0]:.0f}–{WORK_ZONE_MS[1]:.0f} м/с", mean_txt))
    wd = d["wind_direction_100m"].astype(float).to_numpy()
    w = d["p50"].astype(float).to_numpy() + 1e-3
    if len(wd) and np.isfinite(wd).any():
        ok = np.isfinite(wd)
        rad = np.deg2rad(wd[ok])
        mean_dir = float((np.degrees(np.arctan2((w[ok] * np.sin(rad)).sum(), (w[ok] * np.cos(rad)).sum())) + 360) % 360)
        layout = park_layout() if not res.site.transfer else None
        if layout:
            n = upwind_count(mean_dir, layout)
            per_hour = np.array([upwind_count(float(x), layout) > 0 for x in wd[ok]])
            wake_share = float((w[ok] * per_hour).sum() / w[ok].sum())
            meaning = (f"наветренная сторона свободна — потерь в следе не ожидается" if n == 0 else
                       f"{100 * wake_share:.0f} % выработки в зоне следа соседних турбин — возможны потери")
            out.append(("Направление", f"{compass(mean_dir)} {mean_dir:.0f}°, с наветренной стороны {n} турб.", meaning))
        else:
            out.append(("Направление", f"{compass(mean_dir)} {mean_dir:.0f}°", "схемы парка нет — след не оценивается"))
    if res.ens_std is not None and len(res.ens_std):
        t = pd.to_datetime(d["target_time_utc"], utc=True)
        e = res.ens_std.reindex(t).astype(float)
        if e.notna().any():
            m = float(e.mean())
            lo, hi = ENS_STD_TERCILES
            meaning = ("модели согласованы — прогноз ветра надёжнее обычного" if m < lo else
                       "обычное расхождение моделей" if m <= hi else "модели расходятся сильнее обычного — выше неопределённость")
            out.append(("Разброс ансамбля", f"{m:.1f} м/с (GFS/ICON/ECMWF)", meaning))
    a = res.analysis
    rev = (a.get("revision") or {}).get(FARM) if a.get("revision") else None
    if rev:
        sig = rev["mae"] > REVISION_MAE_THRESHOLD
        word = "выше" if rev["bias"] > 0 else "ниже"
        meaning = (f"прогноз погоды пересмотрен: выработка {word} прежней оценки" if sig else "прогноз устойчив к обновлению погоды")
        out.append(("Ревизия к D−1", f"MAE {rev['mae']:.2f}, смещение {rev['bias']:+.2f} доли ном.", meaning))
    m = (a.get("metrics") or {}).get(FARM) or (a.get("metrics") or {}).get(main) or {}
    if m.get("climatology_mean") and m.get("vs_climatology_pct") is not None:
        month = MONTHS_RU[int(pd.Timestamp(d["time_local"].iloc[0]).month) - 1] if len(d) else "месяц"
        pct = float(m["vs_climatology_pct"])
        meaning = ("близко к обычной выработке" if abs(pct) <= 15 else
                   f"выработка {'выше' if pct > 0 else 'ниже'} обычной для месяца")
        out.append((f"Норма ({month})", f"{m['mean_p50']:.2f} против {m['climatology_mean']:.2f} ({pct:+.0f} %)", meaning))
    return out[:6]


# ---------------------------------------------------------------- тестовый период (результаты replay, только чтение)
REPLAY_SUMMARY = config.OUTPUTS_DIR / "replay_summary.csv"
LATEST_BY_TARGET = config.OUTPUTS_DIR / "forecasts" / "latest_by_target.csv"
TEST_START, TEST_END = date.fromisoformat(config.TEST_ISSUE_START), date.fromisoformat(config.TEST_ISSUE_END)


def replay_stamp() -> tuple:
    return tuple(int(p.stat().st_mtime) if p.exists() else 0 for p in (REPLAY_SUMMARY, LATEST_BY_TARGET))


def load_replay() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(сшитый прогноз парка: последний выпуск на каждый час + previous_p50 и revision, сводка выпусков)."""
    stitched = summary = None
    try:
        df = pd.read_csv(LATEST_BY_TARGET)
        df = df[df["turbine"] == FARM].copy()
        df["time_local"] = pd.to_datetime(df["target_time_local"].astype(str).str[:19])
        stitched = df.sort_values("time_local").reset_index(drop=True)
    except Exception:  # noqa: BLE001 — файла может не быть до make replay
        stitched = None
    try:
        summary = pd.read_csv(REPLAY_SUMMARY)
        summary = summary[summary["issue_date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}$")]
        summary = summary.sort_values("issue_date").reset_index(drop=True)
    except Exception:  # noqa: BLE001
        summary = None
    return stitched, summary


# ---------------------------------------------------------------- атлас, оценка площадки, 3D-сцена (модули ядра необязательны)
KZ_GEO = config.ROOT / "data" / "atlas" / "kaz.geo.json"
ATLAS_STEP = 0.5                                # шаг сетки атласа, °
STUDY_START, STUDY_END = "2025-01-01", "2025-12-31"   # год ERA5 для оценки ресурса
STUDY_STEPS = ("Атлас ветра", "ERA5 за год и оценка ресурса", "Прогноз 48 ч переносом модели", "Рельеф и 3D-сцена",
               "Отчёт для руководства")


def _optional(name: str):
    """Модуль ядра wind_agent.<name> или None, если в этой сборке его нет (или он не импортируется)."""
    import importlib
    try:
        return importlib.import_module(f"wind_agent.{name}")
    except Exception:  # noqa: BLE001 — раздел панели покажет «недоступно в этой сборке»
        return None


def backend() -> dict:
    """Какие модули площадочного анализа есть в сборке: atlas, assess, vizdata."""
    return {n: _optional(n) is not None for n in ("atlas", "assess", "vizdata")}


@functools.lru_cache(maxsize=1)
def kz_outline() -> list[list[tuple[float, float]]]:
    """Контур Казахстана [(lon, lat)] — из atlas.kz_polygon() или data/atlas/kaz.geo.json."""
    atlas = _optional("atlas")
    if atlas is not None and hasattr(atlas, "kz_polygon"):
        try:
            return [[(float(x), float(y)) for x, y in atlas.kz_polygon()]]
        except Exception:  # noqa: BLE001
            pass
    if not KZ_GEO.exists():
        return []
    gj = json.loads(KZ_GEO.read_text(encoding="utf-8"))
    rings = []
    for f in gj.get("features", []):
        g = f.get("geometry") or {}
        polys = [g["coordinates"]] if g.get("type") == "Polygon" else g.get("coordinates", [])
        rings += [[(float(x), float(y)) for x, y in p[0]] for p in polys]
    return rings


def in_kz(lat: float, lon: float) -> bool:
    atlas = _optional("atlas")
    if atlas is not None and hasattr(atlas, "in_kazakhstan"):
        try:
            return bool(atlas.in_kazakhstan(lat, lon))
        except Exception:  # noqa: BLE001
            pass
    rings = kz_outline()
    if not rings:
        return KZ_LAT[0] <= lat <= KZ_LAT[1] and KZ_LON[0] <= lon <= KZ_LON[1]
    from matplotlib.path import Path as MPath
    return any(MPath(r).contains_point((lon, lat)) for r in rings)


def atlas_grid() -> dict | None:
    """Ячейки атласа для карты: {"cells": DataFrame[id, lat, lon, ws100_est, ws50_ann, resource_class], "geojson"}.
    None — модуля атласа нет; тогда карта показывает контур и сетку центров без цвета."""
    atlas = _optional("atlas")
    if atlas is None:
        return None
    try:
        gj = atlas.grid_cells_geojson()
        rows = []
        for f in gj.get("features", []):
            p = f.get("properties") or {}
            rows.append({"id": str(f.get("id") or f"{p.get('lat')}_{p.get('lon')}"), "lat": float(p["lat"]),
                         "lon": float(p["lon"]), "ws100_est": float(p.get("ws100_est") or np.nan),
                         "ws50_ann": float(p.get("ws50_ann") or np.nan), "resource_class": str(p.get("resource_class") or "")})
        if not rows:
            return None
        return {"cells": pd.DataFrame(rows), "geojson": gj}
    except Exception:  # noqa: BLE001
        return None


def fallback_centers() -> pd.DataFrame:
    """Центры ячеек 0.5° внутри контура Казахстана (без атласа — для выбора точки кликом)."""
    lats = np.arange(KZ_LAT[0] + ATLAS_STEP / 2, KZ_LAT[1], ATLAS_STEP)
    lons = np.arange(KZ_LON[0] + ATLAS_STEP / 2, KZ_LON[1], ATLAS_STEP)
    pts = [(round(float(a), 2), round(float(o), 2)) for a in lats for o in lons]
    rings = kz_outline()
    if rings:
        from matplotlib.path import Path as MPath
        paths = [MPath(r) for r in rings]
        pts = [(a, o) for a, o in pts if any(p.contains_point((o, a)) for p in paths)]
    return pd.DataFrame(pts, columns=["lat", "lon"])


def nearest_cell(lat: float, lon: float) -> dict | None:
    atlas = _optional("atlas")
    if atlas is None:
        return None
    try:
        return dict(atlas.nearest_cell(float(lat), float(lon)))
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class StudyParams:
    lat: float
    lon: float
    n_turbines: int = 10
    rated_mw: float = 2.5
    name: str = ""

    def key(self) -> tuple:
        return (round(self.lat, 3), round(self.lon, 3), int(self.n_turbines), round(float(self.rated_mw), 2))

    @property
    def label(self) -> str:
        return self.name.strip() or f"Площадка {self.lat:.2f}° с. ш., {self.lon:.2f}° в. д."


def _forecast_summary(res: ForecastResult) -> dict:
    """Короткая сводка прогноза 48 ч для отчёта руководству (числа — из того же анализа, что в панели)."""
    v = res.view(48)
    main = FARM if FARM in set(v["turbine"]) else res.series[0]
    s = series_stats(v, main)
    cap = res.site.capacity_mw(main) or 1.0
    return {"выпуск прогноза (live)": res.issue_label, "часов прогноза": s["hours"],
            "энергия 48 ч P50, МВт·ч": round(s["energy"] * cap, 1), "энергия 48 ч P10, МВт·ч": round(s["energy_p10"] * cap, 1),
            "энергия 48 ч P90, МВт·ч": round(s["energy_p90"] * cap, 1), "средняя загрузка P50, доля ном.": s["mean_p50"],
            "пик P50, МВт": round(s["max_p50"] * cap, 2), "средний ветер 100 м, м/с": s["wind_mean"],
            "решение агента": str(res.final.get("decision") or "—")}


def run_assessment(p: StudyParams, wx: WeatherClient | None = None, progress=None) -> dict:
    """Исследование площадки: атлас → ERA5 за год (assess_site) → прогноз 48 ч (live, перенос модели) →
    рельеф и 3D-сцена (vizdata) → отчёт для руководства (assess.management_report, LLM при наличии ключа).

    progress(i, label, state) — state ∈ {"run", "done", "skip", "error"}. Шаги независимы: ошибка одного
    не останавливает остальные, текст ошибки — в out["errors"][label]."""
    wx = wx or WeatherClient(offline=False)
    out: dict = {"params": p, "cell": None, "assessment": None, "forecast_result": None, "viz_html": None,
                 "report": None, "errors": {}, "skipped": [], "started": time.time()}
    say = progress or (lambda *a: None)
    atlas, assess, vizdata = _optional("atlas"), _optional("assess"), _optional("vizdata")
    site = config.custom_site(p.lat, p.lon, n_turbines=int(p.n_turbines), rated_mw=float(p.rated_mw),
                              name=p.name.strip() or None)

    def sub(i):                                 # сообщения модулей ядра о ходе шага: progress(msg) или progress(k, n, msg)
        return lambda *args: say(i, f"{STUDY_STEPS[i]} · {str(args[-1] if args else '').strip()[:90]}", "run")

    def step(i, fn, available=True):
        label = STUDY_STEPS[i]
        if not available:
            out["skipped"].append(label)
            say(i, label, "skip")
            return None
        say(i, label, "run")
        try:
            r = fn()
            say(i, label, "done")
            return r
        except Exception as e:  # noqa: BLE001 — короткое сообщение в панель, без трейсбека
            out["errors"][label] = f"{type(e).__name__}: {str(e)[:200]}"
            say(i, label, "error")
            return None

    out["cell"] = step(0, lambda: dict(atlas.nearest_cell(p.lat, p.lon)), atlas is not None)
    out["assessment"] = step(1, lambda: assess.assess_site(site, wx, start=STUDY_START, end=STUDY_END, progress=sub(1)),
                             assess is not None)
    fp = ForecastParams(site=CUSTOM_KEY, lat=p.lat, lon=p.lon, n_turbines=int(p.n_turbines), rated_mw=float(p.rated_mw),
                        name=p.name, mode=MODE_LIVE, horizon=48, use_llm=False)
    res = out["forecast_result"] = step(2, lambda: run_forecast(fp, wx=wx, progress=sub(2)))

    def scene():
        payload = vizdata.viz_payload(site, res.forecast if res is not None else None, fetch=True, progress=sub(3))
        return vizdata.viz_html(payload, res.issue_label if res is not None else date.today().isoformat())
    out["viz_html"] = step(3, scene, vizdata is not None)

    def report():
        summary = _forecast_summary(res) if res is not None else None
        return dict(assess.management_report(out["assessment"], forecast_summary=summary, use_llm=llm_available()))
    if assess is not None and out["assessment"] is None:
        out["errors"][STUDY_STEPS[4]] = "не построен: нет оценки ресурса"
        say(4, STUDY_STEPS[4], "error")
    else:
        out["report"] = step(4, report, assess is not None)
    out["elapsed_s"] = round(time.time() - out["started"], 1)
    return out
