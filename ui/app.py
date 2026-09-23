"""WindAgent — панель оператора ВЭС (Streamlit).

Запуск из корня репозитория: `make ui` (или `.venv/bin/streamlit run ui/app.py`).
Вся работа с ядром — в ui/core.py (run_forecast, run_assessment, ask_agent); здесь только ввод параметров и отображение.
Оператор: выпуск → сводка и риски → три показателя → график мощности → экспорт → схема агента.
Чат, объяснения, журналы и 3D открываются по запросу. Все оперативные блоки используют выбранные 24/48 ч.
Исследование новых площадок вынесено в отдельное рабочее пространство.
"""
from __future__ import annotations

import html
import inspect
import json
import re
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:                                    # st.components.v1.html устаревает (1.6x: st.iframe) — держим оба пути
    import streamlit.components.v1 as components
except Exception:  # noqa: BLE001
    components = None

try:                                    # карта Казахстана (extra [ui]); без plotly — только форма координат
    import plotly.graph_objects as go
except Exception:  # noqa: BLE001
    go = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import motion  # noqa: E402

st.set_page_config(page_title="WindAgent — панель оператора ВЭС", layout="wide")

# ---------------------------------------------------------------- совместимость Streamlit 1.38 … 1.6x
_DF_WIDTH = inspect.signature(st.dataframe).parameters.get("width")
_NEW_WIDTH_API = _DF_WIDTH is not None and _DF_WIDTH.default == "stretch"
_toggle = getattr(st, "toggle", st.checkbox)


def _names(fn) -> set[str]:
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def _kw(fn, **kw) -> dict:
    """Только те аргументы, которые есть у функции в установленной версии Streamlit."""
    names = _names(fn)
    return {k: v for k, v in kw.items() if k in names}


def _stretch(fn) -> dict:
    names = _names(fn)
    if _NEW_WIDTH_API and "width" in names:
        return {"width": "stretch"}
    return {"use_container_width": True} if "use_container_width" in names else {}


# ---------------------------------------------------------------- палитра (ISA-101: серая основа, цвет — внимание)
PALETTES = {
    "light": {"band": "#bae6e8", "band_op": 0.45, "p50": "#087f8c", "prev": "#8b95a3", "wind": "#7c9486", "rule": "#9aa3ae",
              "fact": "#111827", "ramp": "#f59e0b", "calm": "#9ca3af", "text": "#4b5563"},
    "dark": {"band": "#207b85", "band_op": 0.25, "p50": "#62d3da", "prev": "#9aa3ae", "wind": "#86a898", "rule": "#6b7280",
             "fact": "#f9fafb", "ramp": "#f59e0b", "calm": "#9ca3af", "text": "#9ca3af"},
}


def palette() -> dict:
    kind = "light"
    try:
        kind = getattr(getattr(st.context, "theme", None), "type", None) or "light"
    except Exception:  # noqa: BLE001 — старые версии без st.context.theme
        pass
    return PALETTES.get(kind, PALETTES["light"])


SITE_LABELS = {core.NURLY_KEY: "ВЭС «Нурлы»", core.CUSTOM_KEY: "Своя площадка"}
MODE_LABELS = {core.MODE_REPLAY: "Ретроспектива", core.MODE_LIVE: "Оперативный"}
UNIT_LABELS = {"mw": "МВт", "frac": "Доля номинала"}
DEC_LEVEL = {"accept": None, "flag": "warn", "recalculate": "err"}
LEVEL_WORD = {"error": "КРИТИЧНО", "warning": "ВНИМАНИЕ", "info": "СВЕДЕНИЯ"}

st.markdown("""<style>
[data-testid="stMainBlockContainer"]{max-width:1260px;padding-top:3.8rem;padding-bottom:2rem}
[data-testid="stHeadingWithActionElements"] h1{font-size:1.7rem;letter-spacing:-.025em;padding:0 0 .3rem;font-weight:650}
[data-testid="stHeadingWithActionElements"] h4{font-size:1rem}
[data-testid="stVerticalBlock"]{gap:.8rem}
[data-testid="stMetricValue"]{font-size:1.3rem}
.wa-context{font-size:.83rem;opacity:.7;margin:0}
.wa-summary{border-left:3px solid #348b95;padding:.2rem 0 .2rem .85rem;margin:.1rem 0 .3rem}
.wa-summary.warn{border-color:#d39a35}.wa-summary.error{border-color:#dc6262}
.wa-summary strong{font-size:1rem;font-weight:600}.wa-summary p{font-size:.88rem;line-height:1.5;margin:.25rem 0 0;opacity:.8}
@media(max-width:700px){[data-testid="stMainBlockContainer"]{padding:3.8rem .9rem 1.2rem}[data-testid="stHeadingWithActionElements"] h1{font-size:1.4rem}}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{white-space:normal!important;overflow:visible!important;text-overflow:clip!important}
.wa-status{display:flex;flex-wrap:wrap;gap:.35rem .4rem;margin:.1rem 0 .25rem 0}
.wa-chip{display:inline-flex;gap:.4rem;align-items:baseline;padding:.1rem .5rem;border:1px solid rgba(128,128,128,.3);
  border-radius:.3rem;font-size:.8rem;line-height:1.55;white-space:nowrap}
.wa-k{opacity:.6}
.wa-chip-warn{border-color:#ca8a04;background:rgba(234,179,8,.14)}
.wa-chip-err{border-color:#dc2626;background:rgba(220,38,38,.12)}
.wa-tech{font-size:.75rem;opacity:.6;margin:0 0 .6rem 0}
.wa-banner{padding:.3rem .7rem;border-left:3px solid #ca8a04;background:rgba(234,179,8,.12);font-size:.85rem;margin:0 0 .7rem}
.wa-alerts{display:flex;flex-direction:column;gap:.25rem;margin:.1rem 0 .9rem}
.wa-alert{padding:.25rem .65rem;border-left:3px solid;font-size:.85rem;line-height:1.45}
.wa-alert b{font-weight:600}
.wa-lvl{font-size:.7rem;font-weight:700;letter-spacing:.05em;margin-right:.5rem}
.wa-dim{opacity:.7}
.wa-alert-error{border-color:#dc2626;background:rgba(220,38,38,.10)}
.wa-alert-error .wa-lvl{color:#dc2626}
.wa-alert-warning{border-color:#ca8a04;background:rgba(234,179,8,.12)}
.wa-alert-warning .wa-lvl{color:#b45309}
.wa-alert-info{border-color:rgba(128,128,128,.5);background:rgba(128,128,128,.08)}
.wa-sub{font-size:.78rem;opacity:.7;margin-top:-.3rem}
.wa-sub-warn{color:#b45309;opacity:1}
.wa-why{display:grid;grid-template-columns:max-content max-content 1fr;gap:.15rem .9rem;font-size:.85rem;margin:.1rem 0 .9rem;align-items:baseline}
.wa-why-k{opacity:.65}
.wa-why-v{font-weight:600;font-variant-numeric:tabular-nums}
.wa-why-m{opacity:.85}
[data-testid="stNumberInputStepDown"],[data-testid="stNumberInputStepUp"]{display:none}
.wa-ph{display:flex;align-items:center;justify-content:center;border:1px solid rgba(128,128,128,.3);border-radius:.4rem;font-size:.85rem;opacity:.75;background:rgba(128,128,128,.06);text-align:center;padding:1rem}
@media (max-width:700px){.wa-why{grid-template-columns:1fr;gap:0}.wa-why-m{margin-bottom:.35rem}}
.st-key-wa_hero{padding:0;margin:.1rem 0 .6rem}
.st-key-wa_hero [data-testid="stChatMessage"]{padding:.35rem .6rem;background:rgba(15,28,46,.55);border:1px solid rgba(56,189,248,.14)}
.st-key-wa_hero [data-testid="stChatInput"]{border-color:rgba(45,212,191,.35)}
.st-key-wa_hero iframe{display:block}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- кэш ресурсов
@st.cache_resource(show_spinner=False)
def weather_client():
    """Один клиент Open-Meteo на процесс: дисковый кэш data/cache/openmeteo, при промахе — запрос к API."""
    return core.WeatherClient(offline=False)


@st.cache_resource(show_spinner=False)
def core_resources() -> dict:
    """Модель мощности и климатология SCADA — загружаются один раз на процесс."""
    return core.load_core()


@st.cache_resource(show_spinner=False)
def scada_actual() -> dict:
    try:
        return core.nurly_actual()
    except Exception:  # noqa: BLE001 — без факта панель работает
        return {}


# ---------------------------------------------------------------- форматирование
def esc(x) -> str:
    return html.escape(str(x))


def series_label(s: str) -> str:
    return "Парк" if s == core.FARM else s.upper()


def num(x: float | None, nd: int = 1) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:,.{nd}f}".replace(",", " ")


def chips(items: list[tuple[str, str, str | None]]) -> None:
    """Строка состояния: [(подпись, значение, уровень None|warn|err)]."""
    body = "".join(f'<span class="wa-chip{" wa-chip-" + lvl if lvl else ""}"><span class="wa-k">{esc(k)}</span>{esc(v)}</span>'
                   for k, v, lvl in items)
    st.markdown(f'<div class="wa-status">{body}</div>', unsafe_allow_html=True)


def metric(col, label: str, value: str, sub: str | None = None, delta: str | None = None, spark=None,
           spark_type: str = "line", warn: bool = False) -> None:
    with col.container(border=True):
        kw = {"delta": delta, "delta_color": "off"} if delta else {}
        if spark is not None and len(spark):
            kw.update(_kw(st.metric, chart_data=[float(x) for x in spark], chart_type=spark_type))
        st.metric(label, value, **kw)
        cls = "wa-sub wa-sub-warn" if warn else "wa-sub"
        st.markdown(f'<div class="{cls}">{esc(sub) if sub else "&nbsp;"}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- боковая панель
def shift_issue(days: int) -> None:
    """Кнопки «← D−1» / «D+1 →»: соседний выпуск тестового периода и сразу расчёт."""
    new = (st.session_state.get("issue_date") or core.REPLAY_DEFAULT) + timedelta(days=days)
    if core.TEST_START <= new <= core.TEST_END:
        st.session_state["issue_date"] = new
        st.session_state["_run_now"] = True


def open_issue(d) -> None:
    """«Открыть выпуск» во вкладке «Тестовый период»: «Нурлы», ретроспектива, 23:00 и расчёт."""
    if d is None:
        return
    for k, v in (("mode", core.MODE_REPLAY), ("issue_date", d), ("issue_hour", 23), ("_run_now", True)):
        st.session_state[k] = v


def sidebar() -> tuple[core.ForecastParams, bool]:
    """Параметры выпуска для ВЭС «Нурлы». Своя площадка задаётся картой и формой на главной (режим live)."""
    caps = core.capabilities()
    with st.sidebar:
        st.markdown("**Выпуск · ВЭС «Нурлы»**")
        mode = st.radio("Режим", list(MODE_LABELS), format_func=MODE_LABELS.get, horizontal=True, key="mode")
        issue_date, issue_hour = core.REPLAY_DEFAULT, 23
        if mode == core.MODE_REPLAY:
            st.session_state.setdefault("issue_date", core.REPLAY_DEFAULT)   # значения задаём через state, а не value=:
            st.session_state.setdefault("issue_hour", 23)                    # их меняют кнопки D−1/D+1 и «Открыть выпуск»
            issue_date = st.date_input("Дата выпуска", min_value=core.REPLAY_MIN, max_value=core.REPLAY_MAX,
                                       key="issue_date", **_kw(st.date_input, format="DD.MM.YYYY"))
            in_test = core.TEST_START <= issue_date <= core.TEST_END
            b1, b2 = st.columns(2)
            b1.button("← D−1", key="issue_prev", on_click=shift_issue, args=(-1,),
                      disabled=not (in_test and issue_date > core.TEST_START), **_stretch(st.button))
            b2.button("D+1 →", key="issue_next", on_click=shift_issue, args=(1,),
                      disabled=not (in_test and issue_date < core.TEST_END), **_stretch(st.button))
            issue_hour = st.selectbox("Час выпуска, UTC+5", list(range(24)), format_func=lambda h: f"{h:02d}:00",
                                      key="issue_hour")
            if issue_hour != 23 and not caps["issue_hour"]:
                st.caption("Текущая версия ядра выпускает прогноз только в 23:00")
        horizon = st.radio("Горизонт", [24, 48], index=1, horizontal=True, format_func=lambda h: f"{h} ч", key="horizon")
        has_key = core.llm_available()
        use_llm = _toggle("AI-пояснения", value=False, disabled=not has_key, key="use_llm",
                          on_change=activate_assistant,
                          help="Агент объясняет результат и отвечает на вопросы. Режим не меняет точность модели прогноза."
                          if has_key else "Для AI-пояснений нужен ключ OpenAI. Прогноз и ответы по правилам доступны.")
        if st.session_state.pop("_assistant_activated", False):
            st.markdown(motion.mode_activation_markup(), unsafe_allow_html=True)
        run = st.button("Рассчитать", type="primary", **_stretch(st.button))

    params = core.ForecastParams(site=core.NURLY_KEY, mode=mode, issue_date=issue_date, issue_hour=int(issue_hour),
                                 horizon=int(horizon), use_llm=bool(use_llm and has_key))
    return params, run


# ---------------------------------------------------------------- запуск
def friendly_error(e: Exception) -> str:
    import requests
    if isinstance(e, requests.RequestException):
        return f"Open-Meteo недоступен, данных в кэше нет ({type(e).__name__})"
    if isinstance(e, RuntimeError) and "offline" in str(e):
        return "Нет данных в кэше для этой даты"
    return f"{type(e).__name__}: {e}"


def activate_assistant() -> None:
    st.session_state["_assistant_activated"] = bool(st.session_state.get("use_llm"))


def run_and_store(params: core.ForecastParams, *, explicit: bool = False) -> None:
    slot = st.empty()
    completed = {}

    def progress(message):
        match = re.match(r"\s*(\w+) \[(ok|error|warning)\]", message)
        if match:
            completed[match[1]] = {"tool": match[1], "status": match[2]}
            slot.markdown(motion.progress_markup(list(completed.values())), unsafe_allow_html=True)

    try:
        slot.markdown(motion.progress_markup(), unsafe_allow_html=True)
        core_resources()
        res = core.run_forecast(params, wx=weather_client(), progress=progress)
        st.session_state["result"] = res
        st.session_state["_animate_result"] = explicit
        st.session_state.pop("error", None)
    except Exception as e:  # noqa: BLE001 — ошибку показываем оператору, панель не падает
        st.session_state["error"] = (friendly_error(e), traceback.format_exc())
    finally:
        slot.empty()


# ---------------------------------------------------------------- шапка
def title_for(params: core.ForecastParams, res: core.ForecastResult | None) -> str:
    if res is not None and res.params.run_key() == params.run_key():
        return res.site.short_name
    if params.site == core.CUSTOM_KEY:
        return params.name.strip() or f"площадка {params.lat:.3f}, {params.lon:.3f}"
    return "ВЭС «Нурлы»"


def issue_local(res: core.ForecastResult) -> pd.Timestamp | None:
    v = str(res.meta.get("issue_time_local") or "")[:16]
    return pd.Timestamp(v) if v else None


def header(res: core.ForecastResult, view: pd.DataFrame, horizon: int, stale: bool, analysis: dict | None = None) -> None:
    iss = issue_local(res)
    release = f"{iss:%d.%m.%Y %H:%M}" if iss is not None else res.issue_label
    period = f"{view['time_local'].min():%d.%m %H:%M} — {view['time_local'].max():%d.%m %H:%M}"
    st.markdown(f'<p class="wa-context">{esc(MODE_LABELS[res.params.mode])} · выпуск {esc(release)} · UTC+5<br>'
                f'Прогноз: {esc(period)} · {horizon} ч</p>', unsafe_allow_html=True)
    if stale:
        st.warning("Параметры изменены. Нажмите «Рассчитать», чтобы обновить прогноз.")


# ---------------------------------------------------------------- KPI


def _common_delta(res: core.ForecastResult, view: pd.DataFrame, series: str, cap: float | None) -> str | None:
    """Δ энергии на общих часах с выпуском D−1 (сутки D+1) — строкой для st.metric(delta=...)."""
    p = res.previous
    if p is None or p.empty or series not in set(p["turbine"]):
        return None
    cur = view[view["turbine"] == series][["target_time_utc", "p50"]]
    j = cur.merge(p[p["turbine"] == series][["target_time_utc", "p50"]], on="target_time_utc", suffixes=("", "_prev"))
    if j.empty:
        return None
    d = float(j["p50"].sum() - j["p50_prev"].sum())
    return f"{d * cap:+.1f} к D−1" if cap else f"{d:+.1f} к D−1"


def kpis(res: core.ForecastResult, view: pd.DataFrame, horizon: int) -> None:
    main = main_series(res)
    d = view[view["turbine"] == main].sort_values("lead_hours")
    stats = core.series_stats(view, main)
    cap = res.site.capacity_mw(main)
    scale = cap or 1.0
    energy_unit, power_unit = ("МВт·ч", "МВт") if cap else ("ч.н.", "доли ном.")
    columns = st.columns(3)
    delta = _common_delta(res, view, main, cap)
    metric(columns[0], f"Выработка за {horizon} ч", f"{num(stats['energy'] * scale)} {energy_unit}",
           f"{delta} {energy_unit} · общие часы" if delta else "Сумма почасового прогноза")
    peak = d.loc[d["p50"].idxmax()] if len(d) else None
    metric(columns[1], "Пик мощности", f"{num(stats['max_p50'] * scale, 2)} {power_unit}",
           f"{peak['time_local']:%d.%m в %H:%M}" if peak is not None else None)
    width = stats['band'] * scale
    metric(columns[2], "Разброс прогноза", f"{num(width, 2)} {power_unit}",
           "Средняя ширина почасовой полосы",
           warn=stats['band'] > core.WIDE_BAND_THRESHOLD)


# ---------------------------------------------------------------- веерный график


def fan_chart(res: core.ForecastResult, d: pd.DataFrame, series: str, k: float, unit: str, pal: dict,
              fact: pd.Series | None) -> alt.LayerChart:
    """Полоса P10–P90, P50 выпуска D, P50 выпуска D−1 (пунктир, общие часы), факт SCADA (точки), ветер 100 м (правая ось),
    линия «выпуск», подписи суток, фоном — окна рамп (янтарь) и штиля (серый)."""
    d = d.sort_values("lead_hours")
    df = pd.DataFrame({"t": d["time_local"].values, "utc": d["target_time_utc"].values, "lead": d["lead_hours"].values,
                       "p10": (d["p10"] * k).values, "p50": (d["p50"] * k).values, "p90": (d["p90"] * k).values,
                       "wind": d["wind_speed_100m"].values})
    cur_name = "Прогноз мощности"
    prev_name = None
    p = res.previous
    if p is not None and not p.empty and series in set(p["turbine"]):
        pr = p[p["turbine"] == series][["target_time_utc", "p50"]].rename(columns={"p50": "prev", "target_time_utc": "utc"})
        df = df.merge(pr, on="utc", how="left")
        df["prev"] = df["prev"] * k
        prev_name = "Предыдущий выпуск" if df["prev"].notna().any() else None
    fact_name = None
    if fact is not None:
        df["fact"] = fact.to_numpy() * k
        fact_name = "Факт SCADA"

    iss = issue_local(res)
    t_min = min(df["t"].min(), iss) if iss is not None else df["t"].min()
    x = alt.X("t:T", title=None, scale=alt.Scale(domain=[t_min.isoformat(), df["t"].max().isoformat()]),
              axis=alt.Axis(format="%d.%m %H:%M", labelAngle=0, labelOverlap=True, grid=False))
    ydom = alt.Scale(domain=[0, k])
    ytitle = "Мощность, МВт" if unit == "mw" else "Мощность, доля номинала"
    names = [cur_name] + ([prev_name] if prev_name else []) + ([fact_name] if fact_name else [])
    colors = [pal["p50"]] + ([pal["prev"]] if prev_name else []) + ([pal["fact"]] if fact_name else [])
    dashes = [[1, 0]] + ([[5, 4]] if prev_name else []) + ([[1, 0]] if fact_name else [])
    color = alt.Color("s:N", title=None, scale=alt.Scale(domain=names, range=colors),
                      legend=alt.Legend(orient="top", direction="horizontal", labelLimit=280, symbolStrokeWidth=2))
    dash = alt.StrokeDash("s:N", title=None, legend=None, scale=alt.Scale(domain=names, range=dashes))

    layers = []
    frac = d["p50"].to_numpy()
    rects = [{"t0": w["t0"], "t1": w["t1"], "kind": "Рампа", "info": f"|ΔP50| {w['value']:.2f} доли ном./ч"}
             for w in core.ramp_windows(d)]
    rects += [{"t0": w["t0"] - pd.Timedelta(minutes=30), "t1": w["t1"] + pd.Timedelta(minutes=30), "kind": "Штиль",
               "info": f"{w['hours']} ч, P50 ≤ {core.CALM_LEVEL:.2f}"} for w in core.calm_windows(d)]
    if rects and len(frac):
        rdf = pd.DataFrame(rects)
        for kind, fill in (("Рампа", pal["ramp"]), ("Штиль", pal["calm"])):   # постоянная заливка: без общей шкалы color
            part = rdf[rdf["kind"] == kind]
            if part.empty:
                continue
            layers.append(alt.Chart(part).mark_rect(opacity=0.22, color=fill).encode(
                x="t0:T", x2="t1:T",
                tooltip=[alt.Tooltip("kind:N", title="Окно"), alt.Tooltip("info:N", title="Значение"),
                         alt.Tooltip("t0:T", title="С", format="%d.%m %H:%M"), alt.Tooltip("t1:T", title="По", format="%H:%M")]))
    layers.append(alt.Chart(df).mark_area(opacity=pal["band_op"], color=pal["band"]).encode(
        x=x, y=alt.Y("p10:Q", title=ytitle, scale=ydom), y2="p90:Q"))
    cols = ["p50"] + (["prev"] if prev_name else [])
    long = df.melt(id_vars=["t"], value_vars=cols, var_name="s", value_name="v").dropna(subset=["v"])
    long["s"] = long["s"].map({"p50": cur_name, "prev": prev_name})
    layers.append(alt.Chart(long).mark_line(strokeWidth=2.4).encode(
        x=x, y=alt.Y("v:Q", scale=ydom, title=ytitle), color=color, strokeDash=dash))
    if fact_name:
        fdf = pd.DataFrame({"t": df["t"], "s": fact_name, "v": df["fact"]}).dropna(subset=["v"])
        layers.append(alt.Chart(fdf).mark_circle(size=16, opacity=0.9).encode(x=x, y=alt.Y("v:Q", scale=ydom), color=color))

    # границы суток и подписи D+1 / D+2, линия выпуска
    days = pd.date_range(df["t"].min().normalize() + pd.Timedelta(days=1), df["t"].max(), freq="D")
    starts = [df["t"].min()] + list(days)
    base_day = pd.Timestamp(res.params.issue_date) if res.params.mode == core.MODE_REPLAY else None
    lab = pd.DataFrame({"t": starts, "y": [k * 0.97] * len(starts),
                        "txt": [(f"D+{(pd.Timestamp(t).normalize() - base_day).days} · {pd.Timestamp(t):%d.%m}"
                                 if base_day is not None else f"{pd.Timestamp(t):%d.%m}") for t in starts]})
    if len(days):
        layers.append(alt.Chart(pd.DataFrame({"t": days})).mark_rule(color=pal["rule"], opacity=0.5).encode(x="t:T"))
    layers.append(alt.Chart(lab).mark_text(align="left", dx=4, fontSize=11, color=pal["text"]).encode(
        x="t:T", y=alt.Y("y:Q", scale=ydom), text="txt:N"))
    if iss is not None:
        idf = pd.DataFrame({"t": [iss], "y": [k * 0.88], "txt": [f"выпуск {iss:%H:%M}"]})
        layers.append(alt.Chart(idf).mark_rule(color=pal["p50"], strokeWidth=1.2, strokeDash=[2, 2]).encode(x="t:T"))
        layers.append(alt.Chart(idf).mark_text(align="left", dx=4, fontSize=10, color=pal["text"]).encode(
            x="t:T", y=alt.Y("y:Q", scale=ydom), text="txt:N"))

    hover = alt.selection_point(fields=["t"], nearest=True, on="mouseover", empty=False)
    tip = [alt.Tooltip("t:T", title="Время", format="%d.%m %H:%M"), alt.Tooltip("lead:Q", title="Упреждение, ч"),
           alt.Tooltip("p10:Q", title="P10", format=".2f"), alt.Tooltip("p50:Q", title="P50", format=".2f"),
           alt.Tooltip("p90:Q", title="P90", format=".2f")]
    if prev_name:
        tip.append(alt.Tooltip("prev:Q", title=prev_name, format=".2f"))
    if fact_name:
        tip.append(alt.Tooltip("fact:Q", title=fact_name, format=".2f"))
    tip.append(alt.Tooltip("wind:Q", title="Ветер 100 м, м/с", format=".1f"))
    layers.append(alt.Chart(df).mark_rule(color=pal["rule"]).encode(
        x=x, opacity=alt.condition(hover, alt.value(0.7), alt.value(0)), tooltip=tip).add_params(hover))

    return alt.layer(*layers).properties(height=280)


def main_series(res: core.ForecastResult) -> str:
    return core.FARM if core.FARM in res.series else res.series[0]


def main_chart(res: core.ForecastResult, view: pd.DataFrame, title: str, key: str = "nurly") -> str:
    site, sel = res.site, main_series(res)
    unit = "mw" if site.capacity_mw(sel) else "frac"
    st.markdown(f"**{esc(title)}**")
    d = view[view["turbine"] == sel]
    k = site.capacity_mw(sel) if unit == "mw" else 1.0
    fact = core.actual_for(view, sel, scada_actual()) if not site.transfer else None
    st.altair_chart(fan_chart(res, d, sel, k, unit, palette(), fact), **_stretch(st.altair_chart))
    st.caption("Полоса — сумма почасовых границ турбин. Покрытие парка отдельно не проверено.")
    return unit


def file_stem(res: core.ForecastResult) -> str:
    return f"{res.site.key}_{res.issue_label}" + ("" if res.params.mode == core.MODE_LIVE else f"T{res.params.issue_hour:02d}")


def why_block(res: core.ForecastResult, view: pd.DataFrame) -> None:
    """«Почему такой прогноз»: показатель — значение — что это значит для выработки (из данных, без LLM)."""
    st.markdown("**Почему такой прогноз**")
    lines = core.why_lines(res, view)
    if lines:
        body = "".join(f'<div class="wa-why-k">{esc(k)}</div><div class="wa-why-v">{esc(v)}</div><div class="wa-why-m">{esc(m)}</div>'
                       for k, v, m in lines)
        st.markdown(f'<div class="wa-why">{body}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- тестовый период
@st.cache_data(show_spinner=False, max_entries=4)
def replay_data(stamp: tuple):
    return core.load_replay()


def stitched_charts(df: pd.DataFrame, k: float, unit: str, pal: dict, window) -> tuple:
    """Сшитый прогноз парка (последний выпуск на каждый час) и столбики ревизии к предыдущему выпуску."""
    d = pd.DataFrame({"t": df["time_local"], "issue": df["issue_date"].astype(str), "lead": df["lead_hours"],
                      "p10": df["p10"] * k, "p50": df["p50"] * k, "p90": df["p90"] * k,
                      "prev": df["previous_p50"] * k, "rev": df["revision"] * k})
    dom = [d["t"].min().isoformat(), d["t"].max().isoformat()]
    x = alt.X("t:T", title=None, scale=alt.Scale(domain=dom), axis=alt.Axis(format="%d.%m", labelAngle=0, labelOverlap=True))
    ytitle = "Мощность, МВт" if unit == "mw" else "Мощность, доля номинала"
    ydom = alt.Scale(domain=[0, k])
    names = ["P50 последнего выпуска", "P50 предыдущего выпуска"]
    color = alt.Color("s:N", title=None, scale=alt.Scale(domain=names, range=[pal["p50"], pal["prev"]]),
                      legend=alt.Legend(orient="top", direction="horizontal", labelLimit=260))
    dash = alt.StrokeDash("s:N", title=None, legend=None, scale=alt.Scale(domain=names, range=[[1, 0], [4, 3]]))
    layers = []
    if window is not None:
        layers.append(alt.Chart(pd.DataFrame({"t0": [window[0]], "t1": [window[1]]})).mark_rect(
            opacity=0.12, color=pal["ramp"]).encode(x="t0:T", x2="t1:T"))
    layers.append(alt.Chart(d).mark_area(opacity=pal["band_op"], color=pal["band"]).encode(
        x=x, y=alt.Y("p10:Q", title=ytitle, scale=ydom), y2="p90:Q"))
    long = d.melt(id_vars=["t"], value_vars=["p50", "prev"], var_name="s", value_name="v").dropna(subset=["v"])
    long["s"] = long["s"].map({"p50": names[0], "prev": names[1]})
    layers.append(alt.Chart(long).mark_line(strokeWidth=1.3).encode(
        x=x, y=alt.Y("v:Q", scale=ydom, title=ytitle), color=color, strokeDash=dash))
    hover = alt.selection_point(fields=["t"], nearest=True, on="mouseover", empty=False)
    layers.append(alt.Chart(d).mark_rule(color=pal["rule"]).encode(
        x=x, opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
        tooltip=[alt.Tooltip("t:T", title="Время", format="%d.%m %H:%M"), alt.Tooltip("issue:N", title="Выпуск"),
                 alt.Tooltip("lead:Q", title="Упреждение, ч"), alt.Tooltip("p10:Q", title="P10", format=".2f"),
                 alt.Tooltip("p50:Q", title="P50", format=".2f"), alt.Tooltip("p90:Q", title="P90", format=".2f"),
                 alt.Tooltip("prev:Q", title="P50 предыдущего", format=".2f"),
                 alt.Tooltip("rev:Q", title="Ревизия", format="+.2f")]).add_params(hover))
    power = alt.layer(*layers).properties(height=260)
    thr = core.REVISION_MAE_THRESHOLD * k
    r = d.dropna(subset=["rev"])
    rev = alt.Chart(r).mark_rule(strokeWidth=1.4).encode(
        x=x, y=alt.Y("rev:Q", title=f"Ревизия, {'МВт' if unit == 'mw' else 'доли ном.'}"), y2=alt.datum(0),
        color=alt.condition(f"abs(datum.rev) > {thr}", alt.value(pal["ramp"]), alt.value(pal["prev"])),
        tooltip=[alt.Tooltip("t:T", title="Время", format="%d.%m %H:%M"), alt.Tooltip("rev:Q", title="Ревизия", format="+.2f")]
    ).properties(height=110)
    return power, rev


def tab_test_period(res: core.ForecastResult, unit: str) -> None:
    stitched, summary = replay_data(core.replay_stamp())
    if stitched is None or stitched.empty or summary is None or summary.empty:
        st.markdown('<div class="wa-alert wa-alert-info">Результатов тестового периода нет: make replay</div>',
                    unsafe_allow_html=True)
        return
    nurly = core.nurly_site()
    cap = nurly.capacity_mw(core.FARM) if unit == "mw" else None
    k, e_unit = (cap, "МВт·ч") if cap else (1.0, "ч.н.")
    dec = summary["decision"].value_counts().to_dict()
    n_rev = int(summary["flags"].fillna("").str.contains("revision").sum())
    chips([("Выпусков", str(len(summary)), None),
           ("Период", f"{stitched['time_local'].min():%d.%m} – {stitched['time_local'].max():%d.%m.%Y}", None),
           ("Энергия парка", f"{num(stitched['p50'].sum() * k)} {e_unit}", None),
           ("Решения", " · ".join(f"{x} {dec.get(x, 0)}" for x in ("accept", "flag", "recalculate")), None),
           ("Существенных ревизий", str(n_rev), "warn" if n_rev else None)])
    window = None
    if not res.site.transfer and res.params.mode == core.MODE_REPLAY:
        v = res.view(48)
        window = (v["time_local"].min(), v["time_local"].max() + pd.Timedelta(hours=1))
    power, rev = stitched_charts(stitched, k, unit, palette(), window)
    st.altair_chart(power, **_stretch(st.altair_chart))
    st.altair_chart(rev, **_stretch(st.altair_chart))

    tbl = pd.DataFrame({
        "Дата выпуска": pd.to_datetime(summary["issue_date"]).dt.date,
        f"Энергия 48 ч, {e_unit}": summary["farm_energy_48h"].astype(float) * k,
        f"D+1, {e_unit}": summary["farm_energy_day1"].astype(float) * k,
        f"D+2, {e_unit}": summary["farm_energy_day2"].astype(float) * k,
        "Ревизия MAE, доли ном.": pd.to_numeric(summary["revision_mae"], errors="coerce"),
        "Флаги": summary["flags"].fillna("").astype(str).str.replace(";", ", "),
        "Статус": summary["status"].astype(str),
        "Решение": summary["decision"].astype(str),
        "LLM": summary["llm_used"].astype(str).str.lower().eq("true"),
    })
    cc = st.column_config
    cfg = {"Дата выпуска": cc.DateColumn(format="DD.MM.YYYY", width=105),
           f"Энергия 48 ч, {e_unit}": cc.NumberColumn(format="%.1f", width=135),
           f"D+1, {e_unit}": cc.NumberColumn(format="%.1f", width=100),
           f"D+2, {e_unit}": cc.NumberColumn(format="%.1f", width=100),
           "Ревизия MAE, доли ном.": cc.NumberColumn(format="%.3f", width=160),
           "Флаги": cc.TextColumn(width="medium"), "Статус": cc.TextColumn(width=75),
           "Решение": cc.TextColumn(width=100), "LLM": cc.CheckboxColumn(width=55)}
    sel_kw = _kw(st.dataframe, on_select="rerun", selection_mode="single-row", key="replay_table", placeholder="—")
    event = st.dataframe(tbl, hide_index=True, column_config=cfg, height=300, **sel_kw, **_stretch(st.dataframe))
    dates = list(tbl["Дата выпуска"])
    chosen = None
    if sel_kw.get("on_select"):
        try:
            rows = list(event.selection.rows)
        except Exception:  # noqa: BLE001
            rows = []
        chosen = dates[rows[0]] if rows else None
    else:                                       # Streamlit без выбора строк — выбор списком
        chosen = st.selectbox("Выпуск", dates, format_func=lambda x: f"{x:%d.%m.%Y}", key="replay_pick")
    st.button(f"Открыть выпуск {chosen:%d.%m.%Y}" if chosen else "Открыть выпуск", key="open_issue",
              disabled=chosen is None, on_click=open_issue, args=(chosen,))


# ---------------------------------------------------------------- вкладки
def tab_alerts(rows: list[dict]) -> None:
    if not rows:
        st.markdown('<div class="wa-alerts"><div class="wa-alert wa-alert-info"><span class="wa-lvl">СВЕДЕНИЯ</span>'
                    'Предупреждений нет</div></div>', unsafe_allow_html=True)
        return
    body = []
    for r in rows:
        parts = [f'<span class="wa-lvl">{LEVEL_WORD.get(r["level"], "")}</span><b>{esc(r["title"])}</b>']
        if r.get("interval"):
            parts.append(f'<span class="wa-dim">{esc(r["interval"])}</span>')
        if r.get("value"):
            parts.append(esc(r["value"]))
        if r.get("action"):
            parts.append(f'<span class="wa-dim">→ {esc(r["action"])}</span>')
        body.append(f'<div class="wa-alert wa-alert-{r["level"]}">{" · ".join(parts)}</div>')
    st.markdown(f'<div class="wa-alerts">{"".join(body)}</div>', unsafe_allow_html=True)


def tab_tables(res: core.ForecastResult, view: pd.DataFrame, horizon: int, unit: str) -> None:
    site = res.site
    scale = site.capacity_mw if unit == "mw" else None
    table = (core.daily_table(view, res.series, series_label, scale, "МВт·ч") if unit == "mw"
             else core.daily_table(view, res.series, series_label))
    cc = st.column_config
    table = table.rename(columns={"Сутки (местные)": "Сутки"})
    st.dataframe(table, hide_index=True, **_stretch(st.dataframe),
                 column_config={c: cc.TextColumn(width="small" if c == "Сутки" else "medium") for c in table.columns})
    with st.expander(f"Почасовой прогноз, {horizon} ч"):
        hourly = core.hourly_table(view, res.series, series_label, scale, "МВт" if unit == "mw" else "доля ном.")
        cfg = {"Местное время": cc.DatetimeColumn(**_kw(cc.DatetimeColumn, format="DD.MM HH:mm", width="small", pinned=True)),
               "Упреждение, ч": cc.NumberColumn(format="%d", width="small")}
        for c in hourly.columns:
            if c in cfg:
                continue
            fmt = ("%.2f" if unit == "mw" else "%.3f") if " P" in f" {c}" and ("P10" in c or "P50" in c or "P90" in c) else \
                ("%.0f" if "°" in c else "%.1f")
            cfg[c] = cc.NumberColumn(format=fmt, width="small")
        st.dataframe(hourly, hide_index=True, height=420, column_config=cfg, **_stretch(st.dataframe))
    stem = file_stem(res)
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Экспорт CSV", data=core.forecast_csv(view), file_name=f"forecast_{stem}_{horizon}h.csv",
                       mime="text/csv", help="Почасовой прогноз для выбранного периода", **_stretch(st.download_button))
    c2.download_button("Отчёт полного выпуска · 48 ч", data=res.report_md.encode("utf-8"), file_name=f"report_{stem}.md",
                       mime="text/markdown", disabled=not res.report_md, help="Отчёт агента",
                       **_stretch(st.download_button))


def tab_agent(res: core.ForecastResult) -> None:
    fin = res.final
    dec = str(fin.get("decision", "—"))
    fact = fin.get("fact_check") or {}
    items = [("Решение", f"{dec} — {core.DECISION_RU.get(dec, '')}", DEC_LEVEL.get(dec)),
             ("По правилам", str(fin.get("rules_decision", "—")), None),
             ("Исполнитель", f"LLM {fin.get('model') or core.llm_model_name()}" if fin.get("llm_used") else "правила", None)]
    if fact:
        bad = len(fact.get("unverified") or [])
        items.append(("Числа в тексте", f"{fact.get('checked', 0)}, не подтверждено {bad}", "warn" if bad else None))
    chips(items)
    st.markdown(fin.get("narrative", ""))
    if fin.get("reasoning"):
        st.markdown(f'<div class="wa-sub" style="margin-top:0">Обоснование: {esc(fin["reasoning"])}</div>',
                    unsafe_allow_html=True)
    if fact.get("note"):
        st.markdown(f'<div class="wa-alerts"><div class="wa-alert wa-alert-warning">{esc(fact["note"])}</div></div>',
                    unsafe_allow_html=True)
    st.markdown("**Трасса агента**")
    cc = st.column_config
    st.dataframe(core.trace_table(res.trace), hide_index=True, **_stretch(st.dataframe),
                 column_config={"Шаг": cc.NumberColumn(format="%d", width="small"),
                                "Инструмент": cc.TextColumn(width="small"), "Кто вызвал": cc.TextColumn(width="small"),
                                "Статус": cc.TextColumn(width="small"),
                                "Время, с": cc.NumberColumn(format="%.2f", width="small"),
                                "Итог": cc.TextColumn(width="large")})
    with st.expander("Отчёт агента"):
        if res.report_md:
            st.markdown(res.report_md)
    if res.core_log:
        with st.expander(f"Журнал ядра ({len(res.core_log)})"):
            for lvl, msg in res.core_log:
                st.text(f"[{lvl}] {msg}")


def tab_fact(res: core.ForecastResult, horizon: int = 48) -> None:
    validation = core.historical_validation(forecast_model_sha256=res.meta.get("model_sha256"))
    st.markdown("**Историческая проверка модели**")
    if validation["status"] == "verified":
        record, manifest = validation["metrics"], validation["manifest"]
        period = manifest["test_period"]
        st.caption(f"{period['start']} — {period['end']}. {validation['message']}")
        overall = record["overall"]
        cols = st.columns(3)
        cols[0].metric("Средняя ошибка, % номинала", f"{100 * overall['model']['mae']:.1f}")
        cols[1].metric("Смещение, % номинала", f"{100 * overall['model']['bias']:+.1f}")
        cols[2].metric("Факт внутри P10–P90, %", f"{overall['coverage_p10_p90_pct']:.1f}")
        rows = [{"Ряд / лаг": name.replace("_lead1", " · 1–24 ч").replace("_lead2", " · 25–48 ч").upper(),
                 "MAE, % номинала": 100 * item["model"]["mae"],
                 "RMSE, % номинала": 100 * item["model"]["rmse"],
                 "Покрытие, %": item["coverage_p10_p90_pct"]} for name, item in record["by"].items()]
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, **_stretch(st.dataframe))
        for limitation in manifest.get("limitations", []):
            st.caption(limitation)
    else:
        st.info(validation["message"])
    st.markdown("**Факт SCADA**")
    act = {} if res.site.transfer else scada_actual()
    up = st.file_uploader("Файл факта (формат data/raw/turbine_*.csv)", type=["csv"], accept_multiple_files=True,
                          key="fact_upload")
    if up:
        try:
            act = core.load_uploaded_actual(up)
            if len(act) > 1:
                act[core.FARM] = pd.concat(act.values(), axis=1, sort=True).mean(axis=1)
        except Exception as e:  # noqa: BLE001
            st.error(f"Файл не прочитан: {type(e).__name__}: {e}")
            return
    if not act:
        st.markdown('<div class="wa-alert wa-alert-info">Факта для площадки нет</div>', unsafe_allow_html=True)
        return
    ev = core.accuracy(res.view(horizon), act)
    if not ev.get("by"):
        last = max(str(s.index.max())[:10] for s in act.values())
        st.markdown(f'<div class="wa-alert wa-alert-info">Факт на часы выпуска отсутствует (факт до {esc(last)})</div>',
                    unsafe_allow_html=True)
        return
    by = pd.DataFrame(ev["by"])
    by["Ряд"] = by["turbine"].map(series_label)
    by["Горизонт"] = by["lead_day"].map({1: "1–24 ч", 2: "25–48 ч"})
    out = by[["Ряд", "Горизонт", "n_hours", "mae", "rmse", "bias", "coverage_p10_p90_pct", "mean_actual", "mean_p50"]].rename(
        columns={"n_hours": "Часов", "mae": "MAE, доли ном.", "rmse": "RMSE, доли ном.", "bias": "Смещение, доли ном.",
                 "coverage_p10_p90_pct": "Покрытие P10–P90, %", "mean_actual": "Факт ср., доли ном.",
                 "mean_p50": "P50 ср., доли ном."})
    cc = st.column_config
    st.dataframe(out, hide_index=True, **_stretch(st.dataframe),
                 column_config={"Ряд": cc.TextColumn(width="small"), "Горизонт": cc.TextColumn(width="small"),
                                "Часов": cc.NumberColumn(format="%d", width="small"),
                                "MAE, доли ном.": cc.NumberColumn(format="%.3f"), "RMSE, доли ном.": cc.NumberColumn(format="%.3f"),
                                "Смещение, доли ном.": cc.NumberColumn(format="%+.3f"),
                                "Покрытие P10–P90, %": cc.NumberColumn(format="%.1f"),
                                "Факт ср., доли ном.": cc.NumberColumn(format="%.3f"),
                                "P50 ср., доли ном.": cc.NumberColumn(format="%.3f")})
    o = ev.get("overall") or {}
    if o:
        st.markdown(f'<div class="wa-tech">турбины: {o["n_hours"]} ч · MAE {o["mae"]:.3f} · RMSE {o["rmse"]:.3f} · '
                    f'смещение {o["bias"]:+.3f} · покрытие P10–P90 {o["coverage_p10_p90_pct"]:.1f} %</div>',
                    unsafe_allow_html=True)


VIZ_DIR = Path(__file__).resolve().parents[1] / "viz"
VIZ_DATA_TAG = '<script src="data/viz_data.js"></script>'


@st.cache_data(show_spinner=False, max_entries=8)
def viz_html(issue_date: str, stamp: tuple) -> str | None:
    """viz/index.html с инлайн-данными viz/data/viz_data.js (srcdoc-iframe не видит относительных путей) и датой
    выпуска панели в window.VIZ_DEFAULT_ISSUE. stamp — mtime файлов, чтобы кэш сбрасывался при пересборке сцены."""
    page, data = VIZ_DIR / "index.html", VIZ_DIR / "data" / "viz_data.js"
    if not page.exists() or not data.exists():
        return None
    htm = page.read_text(encoding="utf-8")
    if VIZ_DATA_TAG not in htm:
        return None
    js = data.read_text(encoding="utf-8").replace("</script", "<\\/script")
    inline = (f"<script>window.VIZ_DEFAULT_ISSUE = {json.dumps(issue_date)};</script>\n"
              f"<script>\n{js}\n</script>")
    htm = htm.replace(VIZ_DATA_TAG, inline, 1)
    # Вкладки Streamlit рендерятся скрытыми: iframe получает нулевой размер, и сцена падает на инициализации.
    # Основной модуль запускаем, только когда iframe стал видимым (importmap действует и на вставленный позже модуль).
    if '<script type="module">' in htm and "</body>" in htm:
        htm = htm.replace('<script type="module">', '<script type="text/x-wa-deferred" id="wa-viz-main">', 1)
        loader = """<script>
(function () {
  var orig = window.__showErr;
  window.__showErr = function (msg) { if (!window.__waStarted) return; if (orig) orig(msg); };
  function start() {
    if (window.__waStarted || window.innerWidth < 50 || window.innerHeight < 50) return;
    window.__waStarted = true;
    var src = document.getElementById('wa-viz-main'), s = document.createElement('script');
    s.type = 'module'; s.textContent = src.textContent; document.body.appendChild(s);
    setTimeout(function () { if (!window.__vizStarted && orig) orig('three.js не загрузился с CDN (cdn.jsdelivr.net).'); }, 8000);
  }
  window.addEventListener('resize', start);
  var iv = setInterval(function () { start(); if (window.__waStarted) clearInterval(iv); }, 250);
  start();
})();
</script>
"""
        htm = htm.replace("</body>", loader + "</body>", 1)
    return htm


TOP_H = 540                             # высота верхнего ряда: 3D-сцена слева, карта с формой справа
MAP_H = 330
NA = "недоступно в этой сборке"


def nurly_scene_html(res: core.ForecastResult) -> str | None:
    stamp = tuple(int(f.stat().st_mtime) if f.exists() else 0 for f in (VIZ_DIR / "index.html", VIZ_DIR / "data" / "viz_data.js"))
    issue = res.params.issue_date.isoformat() if res.params.mode == core.MODE_REPLAY else str(res.issue_label)
    return viz_html(issue, stamp)


def placeholder(text: str, height: int = TOP_H) -> None:
    st.markdown(f'<div class="wa-ph" style="height:{height}px"><span>{esc(text)}</span></div>', unsafe_allow_html=True)


def scene_block(res: core.ForecastResult | None, study: dict | None) -> None:
    """3D-сцена текущей площадки: «Нурлы» — из viz/data/viz_data.js; своя площадка — vizdata после исследования."""
    htm = None
    if study is not None:
        htm = study.get("viz_html")
        if not htm:
            err = study.get("errors", {}).get(core.STUDY_STEPS[3])
            placeholder(f"3D-сцена площадки: {err}" if err else f"3D-сцена площадки: {NA}")
            return
    elif res is not None:
        htm = nurly_scene_html(res)
    if not htm:
        placeholder("Данных 3D-сцены нет: make viz")
        return
    if hasattr(st, "iframe"):
        st.iframe(htm, height=TOP_H)
    elif components is not None and hasattr(components, "html"):
        components.html(htm, height=TOP_H, scrolling=False)


# ---------------------------------------------------------------- карта Казахстана и выбор площадки
@st.cache_data(show_spinner=False, max_entries=2)
def atlas_grid(stamp: tuple) -> dict | None:
    return core.atlas_grid()


@st.cache_data(show_spinner=False, max_entries=1)
def fallback_centers() -> pd.DataFrame:
    return core.fallback_centers()


def atlas_stamp() -> tuple:
    f = core.config.ROOT / "data" / "atlas" / "kz_wind_atlas.csv"
    return (core.backend()["atlas"], int(f.stat().st_mtime) if f.exists() else 0)


# Подложка без внешних тайлов: растровые тайлы CARTO требуют API-ключ, а строковый стиль без сети не загружается
# и карта остаётся пустой. Фон + контур Казахстана + города-ориентиры работают офлайн.
# sources не пустой: plotly выбрасывает пустой dict при сериализации, и MapLibre отвергает стиль без "sources".
MAP_STYLE = {"version": 8, "sources": {"wa-empty": {"type": "geojson", "data": {"type": "Point", "coordinates": [0, 0]}}},
             "layers": [{"id": "bg", "type": "background", "paint": {"background-color": "#11151c"}}]}
CITIES = [("Астана", 51.17, 71.43), ("Алматы", 43.24, 76.89), ("Шымкент", 42.32, 69.59), ("Актобе", 50.28, 57.17),
          ("Атырау", 47.11, 51.92), ("Актау", 43.65, 51.17), ("Караганда", 49.80, 73.10), ("Павлодар", 52.29, 76.97),
          ("Усть-Каменогорск", 49.95, 82.61), ("Костанай", 53.21, 63.62), ("Уральск", 51.23, 51.37),
          ("Кызылорда", 44.85, 65.51), ("Тараз", 42.90, 71.37), ("Талдыкорган", 45.02, 78.37), ("Жезказган", 47.78, 67.71),
          ("Семей", 50.41, 80.25), ("Балхаш", 46.85, 74.98)]


def cand_defaults(grid: dict | None) -> None:
    """Кандидат по умолчанию — ячейка атласа с наибольшим ветром на 100 м."""
    if "cand_lat" in st.session_state:
        return
    lat, lon = 47.0, 52.0
    if grid is not None:
        c = grid["cells"].dropna(subset=["ws100_est"])
        if len(c):
            best = c.loc[c["ws100_est"].idxmax()]
            lat, lon = float(best["lat"]), float(best["lon"])
    st.session_state.update(cand_lat=round(lat, 2), cand_lon=round(lon, 2), cand_n=10, cand_mw=2.5, cand_name="")


def on_map_select() -> None:
    """Клик по карте → координаты центра ячейки в форму (слой центров ячеек отдаёт lat/lon напрямую)."""
    ev = st.session_state.get("kz_map")
    try:
        pts = list(ev["selection"]["points"]) if ev else []
    except Exception:  # noqa: BLE001
        pts = []
    centers = st.session_state.get("_map_centers") or []
    for p in pts:
        lat, lon = p.get("lat"), p.get("lon")
        if (lat is None or lon is None) and p.get("curve_number") == st.session_state.get("_map_cells_curve"):
            i = p.get("point_index", p.get("point_number"))
            if i is not None and 0 <= int(i) < len(centers):
                lat, lon = centers[int(i)]
        if lat is not None and lon is not None:
            st.session_state["cand_lat"], st.session_state["cand_lon"] = round(float(lat), 2), round(float(lon), 2)
            return


def kz_figure(grid: dict | None, cand: tuple[float, float], study: dict | None):
    fig = go.Figure()
    if grid is not None:
        c = grid["cells"]
        fig.add_trace(go.Choroplethmap(
            geojson=grid["geojson"], locations=c["id"], z=c["ws100_est"], featureidkey="id", colorscale="Turbo",
            zmin=float(np.nanpercentile(c["ws100_est"], 2)), zmax=float(np.nanmax(c["ws100_est"])),
            marker_opacity=0.62, marker_line_width=0, hoverinfo="skip",
            colorbar=dict(title=dict(text="Ветер 100 м, м/с", side="right", font=dict(size=11)), thickness=10, len=0.85,
                          x=0.995, xanchor="right", bgcolor="rgba(17,21,28,.6)", tickfont=dict(size=10))))
        centers = c[["lat", "lon"]].to_numpy()
        custom = np.column_stack([c["ws100_est"].round(2), c["ws50_ann"].round(2), c["resource_class"]])
        hover = ("%{lat:.2f}° с. ш., %{lon:.2f}° в. д.<br>ветер 100 м %{customdata[0]} м/с · 50 м %{customdata[1]} м/с"
                 "<br>ресурс: %{customdata[2]}<extra></extra>")
    else:
        fc = fallback_centers()
        centers = fc[["lat", "lon"]].to_numpy()
        custom, hover = None, "%{lat:.2f}° с. ш., %{lon:.2f}° в. д.<extra></extra>"
    for ring in core.kz_outline():
        xs, ys = zip(*ring)
        fig.add_trace(go.Scattermap(lon=list(xs), lat=list(ys), mode="lines", line=dict(color="#cbd5e1", width=1.1),
                                    hoverinfo="skip"))
    fig.add_trace(go.Scattermap(lat=[c[1] for c in CITIES], lon=[c[2] for c in CITIES], text=[c[0] for c in CITIES],
                                mode="markers", marker=dict(size=4, color="#9ca3af"), hovertemplate="%{text}<extra></extra>"))
    st.session_state["_map_centers"] = [(float(a), float(b)) for a, b in centers]
    st.session_state["_map_cells_curve"] = len(fig.data)
    fig.add_trace(go.Scattermap(lat=centers[:, 0], lon=centers[:, 1], mode="markers", customdata=custom,
                                marker=dict(size=15, opacity=0.02, color="#ffffff"), hovertemplate=hover, name="cells"))
    n_lat, n_lon = map(float, np.mean([p[1:] for p in core.nurly_points()], axis=0))
    fig.add_trace(go.Scattermap(lat=[n_lat], lon=[n_lon], mode="markers", marker=dict(size=11, color="#e5e7eb"),
                                hovertemplate="ВЭС «Нурлы»<br>%{lat:.3f}°, %{lon:.3f}°<extra></extra>"))
    if study is not None:
        sp = study["params"]
        fig.add_trace(go.Scattermap(lat=[sp.lat], lon=[sp.lon], mode="markers", marker=dict(size=12, color="#22c55e"),
                                    hovertemplate=f"{esc(sp.label)}<br>исследована<extra></extra>"))
    fig.add_trace(go.Scattermap(lat=[cand[0]], lon=[cand[1]], mode="markers",
                                marker=dict(size=13, color="#f59e0b"),
                                hovertemplate="Кандидат<br>%{lat:.2f}°, %{lon:.2f}°<extra></extra>"))
    fig.update_layout(map=dict(style=MAP_STYLE, center=dict(lat=48.0, lon=67.2), zoom=2.55),
                      height=MAP_H, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, clickmode="event+select",
                      paper_bgcolor="rgba(0,0,0,0)", dragmode="pan", uirevision="kz_map",
                      hoverlabel=dict(bgcolor="#1f2937", font=dict(color="#f9fafb", size=12)))
    return fig


def request_study() -> None:
    st.session_state["_study_req"] = core.StudyParams(
        lat=float(st.session_state["cand_lat"]), lon=float(st.session_state["cand_lon"]),
        n_turbines=int(st.session_state["cand_n"]), rated_mw=float(st.session_state["cand_mw"]),
        name=str(st.session_state.get("cand_name") or ""))


def back_to_nurly() -> None:
    st.session_state["active_study"] = None
    st.session_state["workspace"] = "Оператор"


def map_block(study: dict | None) -> None:
    grid = atlas_grid(atlas_stamp())
    cand_defaults(grid)
    cand = (float(st.session_state["cand_lat"]), float(st.session_state["cand_lon"]))
    if go is not None:
        fig = kz_figure(grid, cand, study)
        cfg = {"displayModeBar": False, "scrollZoom": True}
        kw = _kw(st.plotly_chart, on_select=on_map_select, selection_mode="points", key="kz_map", config=cfg)
        st.plotly_chart(fig, **kw, **_stretch(st.plotly_chart))
    else:
        placeholder(f"Карта Казахстана: plotly {NA}", MAP_H)
    cell = core.nearest_cell(*cand)
    inside = core.in_kz(*cand)
    if not inside:
        line = "точка вне Казахстана"
    elif cell:
        line = (f"ячейка атласа {cell.get('lat', 0):.2f}°, {cell.get('lon', 0):.2f}° · ветер 100 м "
                f"{cell.get('ws100_est', float('nan')):.1f} м/с · ресурс {cell.get('resource_class', '—')} · "
                f"{cell.get('elevation_m', float('nan')):.0f} м")
    else:
        line = f"атлас ветра: {NA}"
    st.markdown(f'<div class="wa-tech" style="margin:.1rem 0 .3rem">{esc(line)}</div>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].number_input("Широта, °", min_value=core.KZ_LAT[0], max_value=core.KZ_LAT[1], step=0.01, format="%.2f",
                      key="cand_lat")
    c[1].number_input("Долгота, °", min_value=core.KZ_LON[0], max_value=core.KZ_LON[1], step=0.01, format="%.2f",
                      key="cand_lon")
    c[2].number_input("Турбин", min_value=1, max_value=200, step=1, key="cand_n")
    c[3].number_input("МВт", min_value=0.5, max_value=10.0, step=0.1, format="%.1f", key="cand_mw")
    st.text_input("Название", key="cand_name", placeholder="Название площадки",
                  **_kw(st.text_input, label_visibility="collapsed"))
    c = st.columns([1.25, 1])
    c[0].button("Исследовать площадку", type="primary", on_click=request_study, disabled=not inside,
                key="study_go", **_stretch(st.button))
    c[1].button("Вернуться к «Нурлы»", on_click=back_to_nurly, disabled=study is None, key="study_back",
                **_stretch(st.button))


# ---------------------------------------------------------------- исследование площадки
def run_study(req: core.StudyParams) -> None:
    """Долгий расчёт с ходом по шагам (st.status); результат — в session_state["studies"][ключ]."""
    studies = st.session_state.setdefault("studies", {})
    key = req.key()
    if key in studies and not studies[key].get("errors"):
        st.session_state["active_study"] = key
        return
    icons = {"run": "…", "done": "✓", "skip": "—", "error": "✕"}
    with st.status(f"Исследование: {req.label}", expanded=True) as box:
        lines = {}
        slot = st.empty()

        def progress(i, label, state):
            suffix = {"skip": f" · {NA}", "error": " · ошибка"}.get(state, "")
            lines[i] = f"{icons.get(state, '')} {i + 1}. {label if state == 'run' else core.STUDY_STEPS[i]}{suffix}"
            slot.markdown("  \n".join(lines[k] for k in sorted(lines)))
        try:
            with st.spinner("Загрузка модели"):
                core_resources()
            out = core.run_assessment(req, wx=weather_client(), progress=progress)
        except Exception as e:  # noqa: BLE001
            box.update(label=f"Исследование не выполнено: {friendly_error(e)}", state="error", expanded=False)
            return
        bad = bool(out["errors"])
        box.update(label=f"Исследование: {req.label} · {out.get('elapsed_s', 0):.0f} с"
                         + (" · с ошибками" if bad else ""), state="error" if bad and not out.get("forecast_result")
                   else "complete", expanded=False)
    studies[key] = out
    st.session_state["active_study"] = key


def _f(x, nd=1, pct=False) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    if pct:
        v = v * 100 if v <= 1.5 else v
        return f"{v:.{nd}f} %"
    return num(v, nd)


def _rose_frame(rose: list[dict]) -> pd.DataFrame:
    """Роза ветров: сектор → угол центра (°, от севера по часовой)."""
    df = pd.DataFrame(rose)
    n = len(df)
    if n == 0:
        return df
    sec = df.get("sector")
    ang = None
    if sec is not None:
        if pd.api.types.is_numeric_dtype(sec):
            v = sec.astype(float)
            ang = v * (360.0 / n) if v.max() < n else v
        elif set(map(str, sec)) <= set(core.COMPASS_RU):
            ang = sec.map(lambda s: core.COMPASS_RU.index(str(s)) * 22.5)
    if "deg" in df and pd.api.types.is_numeric_dtype(df["deg"]):
        ang = df["deg"]
    df["ang"] = ang.astype(float) if ang is not None else np.arange(n) * 360.0 / n
    df["name"] = df["ang"].map(core.compass) if sec is None or pd.api.types.is_numeric_dtype(sec) else sec.astype(str)
    share = df.get("share", pd.Series(np.zeros(n))).astype(float)
    df["share_pct"] = share * 100 if share.max() <= 1.0 else share
    return df


def rose_chart(rose: list[dict], pal: dict) -> alt.Chart | None:
    df = _rose_frame(rose)
    if df.empty:
        return None
    n, R = len(df), 120.0
    half = np.pi / n
    df["t0"], df["t1"] = np.deg2rad(df["ang"]) - half, np.deg2rad(df["ang"]) + half
    df["r"] = np.sqrt(df["share_pct"] / max(df["share_pct"].max(), 1e-9)) * R
    ws = df["ws_mean"].astype(float) if "ws_mean" in df else pd.Series(np.nan, index=df.index)
    df["ws"] = ws
    rings = pd.DataFrame({"r": [R * 0.5, R], "t0": 0.0, "t1": 2 * np.pi})
    base = alt.Chart(rings).mark_arc(filled=False, stroke=pal["rule"], strokeOpacity=0.4).encode(
        theta=alt.Theta("t0:Q", scale=None), theta2="t1:Q", radius=alt.Radius("r:Q", scale=None), radius2=alt.datum(0))
    arcs = alt.Chart(df).mark_arc(stroke="#11151c", strokeWidth=0.6).encode(
        theta=alt.Theta("t0:Q", scale=None), theta2="t1:Q", radius=alt.Radius("r:Q", scale=None), radius2=alt.datum(0),
        color=alt.Color("ws:Q", title="м/с", scale=alt.Scale(scheme="turbo"), legend=alt.Legend(orient="right", gradientLength=90)),
        tooltip=[alt.Tooltip("name:N", title="Румб"), alt.Tooltip("share_pct:Q", title="Доля часов, %", format=".1f"),
                 alt.Tooltip("ws:Q", title="Ветер 100 м, м/с", format=".1f")])
    c0 = R + 16                                  # центр дуг — середина фиксированного холста
    lab = pd.DataFrame({"txt": ["С", "В", "Ю", "З"], "a": [0, 90, 180, 270]})
    lab["x"], lab["y"] = c0 + np.sin(np.deg2rad(lab["a"])) * (R + 9), c0 - np.cos(np.deg2rad(lab["a"])) * (R + 9)
    text = alt.Chart(lab).mark_text(fontSize=11, color=pal["text"]).encode(
        x=alt.X("x:Q", scale=None, axis=None), y=alt.Y("y:Q", scale=None, axis=None), text="txt:N")
    return alt.layer(base, arcs, text).properties(height=2 * c0, width=2 * c0, title="Роза ветров, 100 м")


def demote_headings(md: str, by: int = 2) -> str:
    """Заголовки отчёта на два уровня ниже: внутри панели «#» не должен спорить с заголовком страницы."""
    import re
    return re.sub(r"^(#{1,6})(\s)", lambda m: "#" * min(6, len(m.group(1)) + by) + m.group(2), md, flags=re.M)


def study_section(study: dict) -> None:
    p, a, res = study["params"], study.get("assessment"), study.get("forecast_result")
    pal = palette()
    st.subheader(f"Исследование площадки · {p.label}")
    items = [("Координаты", f"{p.lat:.3f}° с. ш., {p.lon:.3f}° в. д.", None),
             ("Проект", f"{p.n_turbines} × {p.rated_mw:.1f} МВт = {p.n_turbines * p.rated_mw:.1f} МВт", None)]
    cell = study.get("cell")
    if cell:
        items.append(("Атлас", f"{_f(cell.get('ws100_est'))} м/с · {cell.get('resource_class', '—')}", None))
    if a:
        per = a.get("period") or {}
        items.append(("ERA5", f"{per.get('start', '—')} – {per.get('end', '—')} · {per.get('hours', '—')} ч", None))
        if a.get("elevation_m") is not None:
            items.append(("Высота", f"{_f(a.get('elevation_m'), 0)} м", None))
    items.append(("Расчёт", f"{study.get('elapsed_s', 0):.0f} с", "warn" if study.get("errors") else None))
    chips(items)
    notes = [("warning", f"{k}: {v}") for k, v in study.get("errors", {}).items()]
    notes += [("info", f"{k}: {NA}") for k in study.get("skipped", [])]
    if a:
        notes += [("warning", str(w)) for w in (a.get("warnings") or [])]
    if notes:
        st.markdown('<div class="wa-alerts">' + "".join(
            f'<div class="wa-alert wa-alert-{lvl}">{esc(t)}</div>' for lvl, t in notes) + "</div>", unsafe_allow_html=True)

    if a:
        ws, wb, bm = a.get("ws100") or {}, a.get("weibull") or {}, a.get("benchmark") or {}
        c = st.columns(5)
        metric(c[0], "Средний ветер 100 м, м/с", _f(ws.get("mean")),
               f"медиана {_f(ws.get('median'))} · P90 {_f(ws.get('p90'))} · макс. {_f(ws.get('max'))}")
        metric(c[1], "Вейбулл k / c", f"{_f(wb.get('k'), 2)} / {_f(wb.get('c'))}",
               f"штиль {_f(a.get('calm_share'), 0, True)} · шторм {_f(a.get('storm_share'), 1, True)}")
        flh = a.get("full_load_hours")
        metric(c[2], "КИУМ", _f(a.get("cf"), 1, True),
               " · ".join(x for x in (f"{flh} ч полной нагрузки" if flh else "",
                                      f"плотность ×{_f(a.get('density_ratio'), 3)}" if a.get("density_ratio") else "") if x)
               or None)
        cal = ((a.get("scenarios") or {}).get("nurly_calibrated") or {}).get("aep_gwh")
        metric(c[3], f"Годовая выработка {p.n_turbines}×{p.rated_mw:.1f} МВт, ГВт·ч", _f(a.get("aep_gwh"), 1),
               f"на турбину {_f(a.get('aep_per_turbine_gwh'), 2)}" + (f" · по кривой «Нурлы» {_f(cal, 1)}" if cal else ""))
        ratio = bm.get("ratio_to_nurly")
        metric(c[4], "Ветер к ВЭС «Нурлы»", f"×{_f(ratio, 2)}" if ratio is not None else "—",
               f"«Нурлы»: ветер {_f(bm.get('nurly_ws100_mean'))} м/с · КИУМ факт {_f(bm.get('nurly_cf_actual'), 1, True)}",
               warn=ratio is not None and float(ratio) < 1)
        c1, c2, c3 = st.columns([2.2, 1.4, 1.6])
        m = pd.DataFrame(a.get("monthly") or [])
        if not m.empty and "ws100_mean" in m:
            m["name"] = m.get("name_ru", m.get("month", pd.Series(range(1, len(m) + 1))).astype(str))
            m["cf_pct"] = m["cf"].astype(float) * (100 if m["cf"].astype(float).max() <= 1.5 else 1) if "cf" in m else np.nan
            order = list(m["name"])
            x = alt.X("name:N", sort=order, title=None, axis=alt.Axis(labelAngle=0))
            bars = alt.Chart(m).mark_bar(color=pal["band"], opacity=0.9).encode(
                x=x, y=alt.Y("ws100_mean:Q", title="Ветер 100 м, м/с"),
                tooltip=[alt.Tooltip("name:N", title="Месяц"), alt.Tooltip("ws100_mean:Q", title="Ветер, м/с", format=".1f"),
                         alt.Tooltip("cf_pct:Q", title="КИУМ, %", format=".1f")])
            line = alt.Chart(m).mark_line(point=True, color=pal["ramp"], strokeWidth=1.8).encode(
                x=x, y=alt.Y("cf_pct:Q", title="КИУМ, %", axis=alt.Axis(orient="right", grid=False)))
            c1.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(
                height=250, title="Месячный ход: ветер (столбцы) и КИУМ (линия)"), **_stretch(st.altair_chart))
        rc = rose_chart(a.get("rose") or [], pal)
        if rc is not None:
            c2.altair_chart(rc, **({"width": "content"} if _NEW_WIDTH_API else {"use_container_width": False}))
            if a.get("prevailing_sector") is not None:
                c2.markdown(f'<div class="wa-tech">преобладающий румб: {esc(a["prevailing_sector"])}</div>',
                            unsafe_allow_html=True)
        dd = pd.DataFrame(a.get("diurnal") or [])
        if not dd.empty and {"hour_local", "ws100_mean"} <= set(dd.columns):
            c3.altair_chart(alt.Chart(dd).mark_line(point=True, color=pal["p50"], strokeWidth=1.6).encode(
                x=alt.X("hour_local:Q", title="Час, UTC+5", scale=alt.Scale(domain=[0, 23]), axis=alt.Axis(tickCount=8)),
                y=alt.Y("ws100_mean:Q", title="Ветер 100 м, м/с", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("hour_local:Q", title="Час"), alt.Tooltip("ws100_mean:Q", title="м/с", format=".1f")]
            ).properties(height=250, title="Суточный ход ветра"), **_stretch(st.altair_chart))
        src = [str(s) for s in (a.get("data_sources") or [])]
        if a.get("power_curve_source"):
            src.append(f"кривая мощности: {a['power_curve_source']}")
        if src:
            st.markdown(f'<div class="wa-tech">{esc(" · ".join(src))}</div>', unsafe_allow_html=True)

    if res is not None:
        v = res.view(48)
        sel = main_series(res)
        s = core.series_stats(v, sel)
        cap = res.site.capacity_mw(sel) or (p.n_turbines * p.rated_mw)
        chips([("LIVE", f"выпуск {res.issue_label}", None), ("Энергия 48 ч", f"{num(s['energy'] * cap)} МВт·ч "
               f"[{num(s['energy_p10'] * cap)}–{num(s['energy_p90'] * cap)}]", None),
               ("Пик P50", f"{s['max_p50'] * cap:.1f} МВт", None), ("Средняя загрузка", f"{100 * s['mean_p50']:.0f} %", None),
               ("Ветер 100 м", f"{s['wind_mean']:.1f} м/с", None), ("Уверенность", core.confidence_word(s["band"]), None)])
        main_chart(res, v, "Прогноз 48 ч · теоретическая выработка площадки (перенос модели «Нурлы»)", key="study")

    rep = study.get("report")
    if rep and rep.get("markdown"):
        c1, c2 = st.columns([5, 1], **_kw(st.columns, vertical_alignment="bottom"))
        c1.markdown("**Отчёт для руководства**")
        stem = f"site_{p.lat:.2f}_{p.lon:.2f}_{p.n_turbines}x{p.rated_mw:.1f}".replace(".", "p")
        c2.download_button("Скачать .md", data=str(rep["markdown"]).encode("utf-8"), file_name=f"report_{stem}.md",
                           mime="text/markdown", key="dl_study_md", **_stretch(st.download_button))
        fc = rep.get("fact_check") or {}
        if rep.get("llm_used"):
            tech = ["LLM " + core.llm_model_name()]
        elif fc.get("llm_unverified"):
            tech = [f"текст LLM отклонён сверкой ({len(fc['llm_unverified'])} неподтв. чисел) — шаблон по данным"]
        else:
            tech = ["LLM не использовалась — шаблон по данным"]
        if fc:
            tech.append(f"сверка чисел: проверено {fc.get('checked', '—')}, не подтверждено {len(fc.get('unverified') or [])}")
        st.markdown(f'<div class="wa-tech">{esc(" · ".join(tech))}</div>', unsafe_allow_html=True)
        with st.container(border=True, **_kw(st.container, height=520)):
            st.markdown(demote_headings(str(rep["markdown"])))
    elif core.STUDY_STEPS[4] in study.get("skipped", []):
        pass



# ---------------------------------------------------------------- карточка агента (герой панели) и диалог
CARD_H = 108
CHAT_KEEP = 6


def _card_payload(res: core.ForecastResult | None, study: dict | None, ws: str = "", horizon: int = 48) -> dict | None:
    try:
        if study is not None:
            return core.study_card(study)
        if ws == "Исследование площадки":
            lat, lon = float(st.session_state.get("cand_lat", 47.0)), float(st.session_state.get("cand_lon", 52.0))
            return core.idle_study_card(lat, lon, int(st.session_state.get("cand_n", 10)),
                                        float(st.session_state.get("cand_mw", 2.5)), core.nearest_cell(lat, lon),
                                        core.in_kz(lat, lon))
        if res is not None:
            return core.forecast_card(res, horizon)
    except Exception:  # noqa: BLE001 — карточка не должна ронять панель
        return None
    return None


def _card_html(payload: dict) -> str:
    """Only an explicit completed run gets a finite acknowledgement; first load stays still."""
    import secrets
    animate = bool(st.session_state.pop("_animate_result", False))
    if animate:
        st.session_state["_run_nonce"] = secrets.token_hex(6)
    data = dict(payload, theme="dark" if palette() is PALETTES["dark"] else "light")
    return core.agent_card_html(data, animate=animate, intro=False,
                                nonce=st.session_state.get("_run_nonce", "initial"), height=CARD_H)


def chat_context_key(res: core.ForecastResult | None, study: dict | None, horizon: int = 48) -> str:
    if study is not None:
        forecast = study.get("forecast_result")
        revision = forecast.meta.get("input_hash", "") if forecast is not None else study.get("elapsed_s", "")
        return "study:" + repr(study["params"].key()) + ":" + str(revision)
    if res is None:
        return f"fc:empty:{horizon}"
    revision = (res.meta.get("issue_time_utc"), res.meta.get("input_hash"))
    return "fc:" + repr((res.params.run_key(), revision, horizon))


def agent_chat(res: core.ForecastResult | None, study: dict | None, horizon: int = 48) -> None:
    """Диалог с агентом: история (не больше 6 реплик) над полем ввода; ответ — core.ask_agent."""
    chats = st.session_state.setdefault("agent_chat", {})
    ck = chat_context_key(res, study, horizon)
    hist = chats.setdefault(ck, [])
    box = st.container()
    q = st.chat_input("Вопрос агенту о прогнозе или площадке", key="agent_q")
    with box:
        for h in hist[-CHAT_KEEP:]:
            with st.chat_message(h["role"]):
                st.markdown(h["content"])
                if h.get("meta"):
                    st.markdown(f'<div class="wa-tech" style="margin:0">{esc(h["meta"])}</div>', unsafe_allow_html=True)
        if q:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Агент сверяется с данными"):
                    try:
                        out = core.ask_agent(q, res, study, hist, horizon=horizon, use_llm=bool(st.session_state.get("use_llm")))
                    except Exception as e:  # noqa: BLE001
                        out = {"answer": f"Ответ не получен: {type(e).__name__}", "llm_used": False, "fact_check": None}
                st.markdown(out["answer"])
                fc = out.get("fact_check") or {}
                meta = [f"LLM {out.get('model') or core.llm_model_name()}" if out.get("llm_used") else "правила"]
                if fc.get("checked"):
                    meta.append(f"проверено чисел {fc['checked']}, не подтверждено {len(fc.get('unverified') or [])}")
                if fc.get("note"):
                    meta.append(str(fc["note"]))
                meta_txt = " · ".join(meta)
                st.markdown(f'<div class="wa-tech" style="margin:0">{esc(meta_txt)}</div>', unsafe_allow_html=True)
            hist += [{"role": "user", "content": q}, {"role": "assistant", "content": out["answer"], "meta": meta_txt}]
            del hist[:-CHAT_KEEP]


def hero_card(res: core.ForecastResult | None, study: dict | None, horizon: int = 48, ws: str = "") -> None:
    payload = _card_payload(res, study, ws, horizon)
    if payload is not None and hasattr(st, "iframe"):
        st.iframe(_card_html(payload), height=CARD_H)
    elif payload is not None and components is not None and hasattr(components, "html"):
        components.html(_card_html(payload), height=CARD_H, scrolling=False)


# ---------------------------------------------------------------- страница


def show_error(err) -> None:
    st.error(err[0])


def page_header(res: core.ForecastResult | None, study: dict | None, ws: str = "") -> None:
    name = study["params"].label if study else ("Исследование площадки" if ws == WS_STUDY else "ВЭС «Нурлы»")
    st.title(f"WindAgent · {name}")


WS_OPER, WS_STUDY = "Оператор", "Исследование площадки"


def workspace_switch() -> str:
    """Верхний переключатель рабочих пространств: «Оператор» (выпуск «Нурлы») и «Исследование площадки»."""
    st.session_state.setdefault("workspace", WS_OPER)
    seg = getattr(st, "segmented_control", None)
    if seg is not None:
        val = seg("Рабочее пространство", [WS_OPER, WS_STUDY], key="workspace",
                  **_kw(seg, required=True, label_visibility="collapsed", width="stretch"))
    else:
        val = st.radio("Рабочее пространство", [WS_OPER, WS_STUDY], key="workspace", horizontal=True,
                       **_kw(st.radio, label_visibility="collapsed"))
    return val or WS_OPER


def changes_block(res: core.ForecastResult, view: pd.DataFrame, a: dict) -> None:
    """Изменения к прошлому выпуску на общих часах выбранного горизонта: энергия, MAE, смещение, наибольшее изменение."""
    p = res.previous
    sel = main_series(res)
    if p is None or p.empty or sel not in set(p["turbine"]):
        return
    cur = view[view["turbine"] == sel][["target_time_utc", "time_local", "p50"]]
    j = cur.merge(p[p["turbine"] == sel][["target_time_utc", "p50"]], on="target_time_utc", suffixes=("", "_prev"))
    if j.empty:
        return
    cap = res.site.capacity_mw(sel) or 1.0
    unit = "МВт·ч" if res.site.capacity_mw(sel) else "ч.н."
    e = j["p50"] - j["p50_prev"]
    i = int(e.abs().to_numpy().argmax())
    rev = (a.get("revision") or {}).get(sel) or {}
    sig = rev.get("mae", 0) > core.REVISION_MAE_THRESHOLD
    st.markdown(f"**Изменения к выпуску {esc(res.previous_issue)}**")
    chips([("Общих часов", str(len(j)), None),
           ("Энергия", f"{float(j['p50'].sum() - j['p50_prev'].sum()) * cap:+.1f} {unit}", None),
           ("MAE", f"{rev.get('mae', float(e.abs().mean())):.2f} доли ном. · порог {core.REVISION_MAE_THRESHOLD:.2f}",
            "warn" if sig else None),
           ("Смещение", f"{rev.get('bias', float(e.mean())):+.2f} доли ном.", None),
           ("Наибольшее", f"{float(e.iloc[i]) * cap:+.2f} {'МВт' if unit == 'МВт·ч' else 'доли ном.'} · "
                          f"{pd.Timestamp(j['time_local'].iloc[i]):%d.%m %H:%M}", None)])


def actions_row(res: core.ForecastResult, view: pd.DataFrame, horizon: int) -> None:
    stem = file_stem(res)
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Экспорт CSV", data=core.forecast_csv(view), file_name=f"forecast_{stem}_{horizon}h.csv",
                       mime="text/csv", key="act_csv", help="Почасовой прогноз для выбранного периода", **_stretch(st.download_button))
    c2.download_button("Отчёт 48 ч", data=res.report_md.encode("utf-8"), file_name=f"report_{stem}.md",
                       mime="text/markdown", key="act_md", disabled=not res.report_md, **_stretch(st.download_button))


def operator_space(res: core.ForecastResult | None, params: core.ForecastParams) -> None:
    err = st.session_state.get("error")
    if res is None:
        if err:
            show_error(err)
        return
    h = params.horizon
    view = res.view(h)
    analysis = core.horizon_analysis(res, h)
    header(res, view, h, stale=res.params.run_key() != params.run_key(), analysis=analysis)
    if err:
        show_error(err)
        st.warning("Новый расчёт не выполнен. Ниже показан предыдущий результат.")
    rows = core.alert_rows(res, view, False, analysis=analysis)
    important = [r for r in rows if r["level"] in ("warning", "error")
                 or r.get("code") in ("low_confidence", "ensemble_gap", "input_quality")]
    important.sort(key=lambda r: {"error": 0, "warning": 1, "info": 2}.get(r["level"], 3))
    summary = core.operator_summary(res, h)
    css = {"warning": "warn", "error": "error"}.get(summary["status"], "")
    st.markdown(f'<div class="wa-summary {css}"><strong>{esc(summary["title"])}</strong>'
                f'<p>{esc(summary["message"])}</p></div>', unsafe_allow_html=True)
    if important:
        tab_alerts(important[:1])
        if len(important) > 1:
            with st.expander(f"Ещё предупреждения · {len(important) - 1}"):
                tab_alerts(important[1:])
    kpis(res, view, h)
    unit = main_chart(res, view, "Прогноз мощности парка")
    actions_row(res, view, h)
    hero_card(res, None, h, WS_OPER)
    with st.expander("Спросить агента", **_kw(st.expander, key="agent_chat_exp")):
        agent_chat(res, None, h)
    with st.expander("Почему такой прогноз и что изменилось"):
        why_block(res, view)
        changes_block(res, view, analysis)
        if rows:
            st.markdown("**Все события выбранного периода**")
            tab_alerts(rows)
    with st.expander("Данные и проверка качества"):
        # Mutually exclusive detail views avoid loading charts, files and 3D on every rerun.
        section = st.radio("Раздел", ["Почасовые данные", "Тестовый период", "Факт и точность", "Журнал расчёта", "3D площадки"],
                           horizontal=True, key="detail_section", label_visibility="collapsed")
        if section == "Почасовые данные":
            tab_tables(res, view, h, unit)
        elif section == "Тестовый период":
            tab_test_period(res, unit)
        elif section == "Факт и точность":
            tab_fact(res, h)
        elif section == "Журнал расчёта":
            st.caption(f"Технический журнал полного выпуска {int(res.forecast['lead_hours'].max())} ч. "
                       "Решение правил не является разрешением на диспетчерское планирование.")
            tab_agent(res)
        elif section == "3D площадки":
            scene_block(res, None)


def study_space(res: core.ForecastResult | None, study: dict | None, req) -> None:
    """«Исследование площадки»: 3D-сцена площадки и карта атласа с формой, ход исследования, результаты."""
    left, right = st.columns([3, 2], gap="medium")
    with left:
        scene_block(res, study)
    with right:
        map_block(study)
    if req is not None:
        # Ход исследования — под сценой и картой; готовый результат — перезапуском: карточка агента и 3D-сцена
        # уже новой площадки. Если исследование упало целиком — статус с ошибкой остаётся.
        run_study(req)
        if st.session_state.get("active_study") == req.key():
            st.rerun()
    if study:
        study_section(study)


def main() -> None:
    params, run = sidebar()
    run = run or bool(st.session_state.pop("_run_now", False))
    if run or ("result" not in st.session_state and "error" not in st.session_state and not params.use_llm):
        run_and_store(params, explicit=run)          # первый заход — сразу расчёт «Нурлы» по параметрам по умолчанию (из кэша, < 1 с)

    res: core.ForecastResult | None = st.session_state.get("result")
    req = st.session_state.pop("_study_req", None)
    studies = st.session_state.setdefault("studies", {})
    study = studies.get(st.session_state.get("active_study")) if st.session_state.get("active_study") else None
    ws = st.session_state.get("workspace") or WS_OPER
    page_header(res, study if ws == WS_STUDY else None, ws)
    ws = workspace_switch()
    if ws == WS_STUDY:
        study_space(res, study, req)
        hero_card(res, study, 48, ws)
        with st.expander("Спросить агента о площадке"):
            if study is None:
                st.caption("Сначала исследуйте выбранную площадку: агенту нужны её данные.")
            else:
                agent_chat(None, study, 48)
    else:
        operator_space(res, params)


main()
