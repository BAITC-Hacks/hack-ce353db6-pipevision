"""WindAgent — панель оператора ВЭС (Streamlit).

Запуск из корня репозитория: `make ui` (или `.venv/bin/streamlit run ui/app.py`).
Вся работа с ядром — в ui/core.py (run_forecast); здесь только ввод параметров и отображение.
Каркас: шапка → 5 KPI → веерный график → вкладки «Предупреждения», «Таблицы и выгрузка», «Агент», «Площадка»,
«Факт и точность», «3D-сцена» (viz/index.html). Пороги рамп, штиля и уверенности — из wind_agent.agent.tools (как у агента).
"""
from __future__ import annotations

import html
import inspect
import json
import sys
import traceback
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:                                    # st.components.v1.html устаревает (1.6x: st.iframe) — держим оба пути
    import streamlit.components.v1 as components
except Exception:  # noqa: BLE001
    components = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

st.set_page_config(page_title="WindAgent — панель оператора ВЭС", layout="wide")

# ---------------------------------------------------------------- совместимость Streamlit 1.38 … 1.6x
_DF_WIDTH = inspect.signature(st.dataframe).parameters.get("width")
_NEW_WIDTH_API = _DF_WIDTH is not None and _DF_WIDTH.default == "stretch"
_toggle = getattr(st, "toggle", st.checkbox)
_toast = getattr(st, "toast", None)


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
    "light": {"band": "#b9c3d1", "band_op": 0.55, "p50": "#1f2937", "prev": "#8b95a3", "wind": "#7c9486", "rule": "#9aa3ae",
              "fact": "#111827", "ramp": "#f59e0b", "calm": "#9ca3af", "text": "#4b5563"},
    "dark": {"band": "#566579", "band_op": 0.55, "p50": "#e5e7eb", "prev": "#9aa3ae", "wind": "#86a898", "rule": "#6b7280",
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
[data-testid="stMetricValue"]{font-size:1.3rem}
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
def sidebar() -> tuple[core.ForecastParams, bool]:
    caps = core.capabilities()
    with st.sidebar:
        site = st.radio("Площадка", list(SITE_LABELS), format_func=SITE_LABELS.get, key="site",
                        help="«Нурлы»: T1, T2, модель обучена на SCADA. Своя площадка: перенос модели.")
        custom = site == core.CUSTOM_KEY
        lat, lon, n_turb, rated, name = 48.0, 68.0, 1, 2.5, ""
        if custom:
            c1, c2 = st.columns(2)
            lat = c1.number_input("Широта, °", min_value=core.KZ_LAT[0], max_value=core.KZ_LAT[1], value=48.0, step=0.1,
                                  format="%.3f", key="lat")
            lon = c2.number_input("Долгота, °", min_value=core.KZ_LON[0], max_value=core.KZ_LON[1], value=68.0, step=0.1,
                                  format="%.3f", key="lon")
            c3, c4 = st.columns(2)
            n_turb = int(c3.number_input("Турбин, шт.", min_value=1, max_value=50, value=1, step=1, key="n_turb"))
            rated = float(c4.number_input("Номинал, МВт", min_value=0.5, max_value=8.0, value=2.5, step=0.1,
                                          format="%.1f", key="rated"))
            name = st.text_input("Название", value="", key="site_name")
            if not caps["site"]:
                st.caption("Своя площадка не поддерживается текущей версией ядра")
        mode = st.radio("Режим", list(MODE_LABELS), format_func=MODE_LABELS.get, horizontal=True, key="mode",
                        help="Ретроспектива — архивный прогноз погоды на момент выпуска; оперативный — последний запуск.")
        issue_date, issue_hour = core.REPLAY_DEFAULT, 23
        if mode == core.MODE_REPLAY:
            issue_date = st.date_input("Дата выпуска", value=core.REPLAY_DEFAULT, min_value=core.REPLAY_MIN,
                                       max_value=core.REPLAY_MAX, key="issue_date", **_kw(st.date_input, format="DD.MM.YYYY"))
            issue_hour = st.selectbox("Час выпуска", list(range(24)), index=23, format_func=lambda h: f"{h:02d}:00",
                                      key="issue_hour", help="Местное время, UTC+5")
            if issue_hour != 23 and not caps["issue_hour"]:
                st.caption("Текущая версия ядра выпускает прогноз только в 23:00")
        horizon = st.radio("Горизонт", [24, 48], index=1, horizontal=True, format_func=lambda h: f"{h} ч", key="horizon")
        has_key = core.llm_available()
        use_llm = _toggle("LLM", value=False, disabled=not has_key, key="use_llm",
                          help=f"OpenAI {core.llm_model_name()}, tool calling" if has_key else "OPENAI_API_KEY не задан")
        run = st.button("Рассчитать", type="primary", **_stretch(st.button))

    params = core.ForecastParams(site=site, lat=float(lat), lon=float(lon), n_turbines=int(n_turb), rated_mw=float(rated),
                                 name=name, mode=mode, issue_date=issue_date, issue_hour=int(issue_hour),
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


def run_and_store(params: core.ForecastParams) -> None:
    try:
        with st.spinner("Загрузка модели"):
            core_resources()
        step = st.empty()
        with st.spinner("Расчёт" + (" (LLM)" if params.use_llm else "")):
            res = core.run_forecast(params, wx=weather_client(), progress=lambda m: step.caption(m.strip()[:160]))
        step.empty()
        st.session_state["result"] = res
        st.session_state.pop("error", None)
        warn = [a for a in core.alert_rows(res, res.view(params.horizon), params.site == core.CUSTOM_KEY)
                if a["level"] in ("warning", "error")]
        st.session_state["toast"] = ("Предупреждений: " + str(len(warn)) + " — " + "; ".join(a["title"] for a in warn[:3])
                                     if warn else None)
    except Exception as e:  # noqa: BLE001 — ошибку показываем оператору, панель не падает
        st.session_state["error"] = (friendly_error(e), traceback.format_exc())


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


def header(res: core.ForecastResult, view: pd.DataFrame, horizon: int, stale: bool) -> None:
    fin, site, a = res.final, res.site, res.analysis
    src = str(res.meta.get("source") or res.forecast["weather_source"].iloc[0])
    dec = str(fin.get("decision", "—"))
    live = res.params.mode == core.MODE_LIVE
    iss = issue_local(res)
    items = [("LIVE" if live else "РЕТРО", f"выпуск {iss:%d.%m.%Y %H:%M} UTC+5" if iss is not None else res.issue_label, None),
             ("Горизонт", f"{view['time_local'].min():%d.%m %H:%M} – {view['time_local'].max():%d.%m %H:%M} · {horizon} ч", None),
             ("Площадка", f"{site.n_turbines} × {site.rated_mw:.1f} МВт" if site.rated_mw else f"{site.n_turbines} турб.", None),
             ("Погода", "Open-Meteo " + ("Previous Runs" if "previous" in src else "Forecast"), None),
             ("Модель", str(res.forecast["model_version"].iloc[0]), None),
             ("Решение агента", f"{dec} · {'LLM' if fin.get('llm_used') else 'правила'}", DEC_LEVEL.get(dec))]
    if stale:
        items.append(("Параметры", "изменены — «Рассчитать»", "warn"))
    chips(items)
    tech = [f"расчёт {res.elapsed_s:.1f} с", f"запросов к Open-Meteo {res.api_calls}", f"статус анализа {a.get('status', '—')}"]
    if a.get("revision"):
        tech.append(f"ревизия к выпуску {a['revision']['previous_issue_date']}")
    st.markdown(f'<div class="wa-tech">{esc(" · ".join(tech))}</div>', unsafe_allow_html=True)


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
    site = res.site
    main = core.FARM if core.FARM in res.series else res.series[0]
    d = view[view["turbine"] == main].sort_values("lead_hours")
    s = core.series_stats(view, main)
    cap = site.capacity_mw(main)
    k = cap or 1.0
    e_unit, p_unit = ("МВт·ч", "МВт") if cap else ("ч.н.", "доли ном.")
    live = res.params.mode == core.MODE_LIVE

    c = st.columns(5)
    metric(c[0], f"Энергия {horizon} ч, {e_unit}", num(s["energy"] * k),
           f"P10–P90 {num(s['energy_p10'] * k)}–{num(s['energy_p90'] * k)}" + (f" · {s['energy']:.1f} ч.н." if cap else ""),
           spark=d["p50"] * k)
    days = f"{num(s['day1'] * k)} / {num(s['day2'] * k) if s['day2'] is not None else '—'}"
    daily = [s["day1"] * k] + ([s["day2"] * k] if s["day2"] is not None else [])
    metric(c[1], ("Сутки 1 / 2, " if live else "D+1 / D+2, ") + e_unit, days,
           f"выпуск D−1: {res.previous_issue}" if res.previous is not None else None,
           delta=_common_delta(res, view, main, cap), spark=daily, spark_type="bar")
    i_peak = int(d["p50"].to_numpy().argmax()) if len(d) else 0
    metric(c[2], f"Пик P50, {p_unit}", f"{s['max_p50'] * k:.2f}",
           f"{pd.Timestamp(d['time_local'].iloc[i_peak]):%d.%m %H:%M}" + (f" · {s['max_p50']:.2f} доли ном." if cap else ""),
           spark=d["p50"] * k)
    jumps = np.abs(np.diff(d["p50"].to_numpy())) if len(d) > 1 else np.array([0.0])
    i_r = int(jumps.argmax())
    ramp = float(jumps[i_r])
    over = ramp > core.RAMP_THRESHOLD
    metric(c[3], f"Риск рампы, {p_unit}/ч", f"{ramp * k:.2f}",
           f"{ramp:.2f} доли ном./ч · порог {core.RAMP_THRESHOLD:.2f} · {pd.Timestamp(d['time_local'].iloc[i_r + 1]):%d.%m %H:%M}"
           if len(d) > 1 else None, spark=jumps * k, spark_type="bar", warn=over)
    metric(c[4], "Уверенность", core.confidence_word(s["band"]),
           f"P90−P10 {s['band']:.2f} доли ном. · порог {core.WIDE_BAND_THRESHOLD:.2f}",
           spark=(d["p90"] - d["p10"]).to_numpy(), spark_type="area", warn=s["band"] > core.WIDE_BAND_THRESHOLD)


# ---------------------------------------------------------------- веерный график
def fan_chart(res: core.ForecastResult, d: pd.DataFrame, series: str, k: float, unit: str, pal: dict,
              fact: pd.Series | None) -> alt.LayerChart:
    """Полоса P10–P90, P50 выпуска D, P50 выпуска D−1 (пунктир, общие часы), факт SCADA (точки), ветер 100 м (правая ось),
    линия «выпуск», подписи суток, фоном — окна рамп (янтарь) и штиля (серый)."""
    d = d.sort_values("lead_hours")
    df = pd.DataFrame({"t": d["time_local"].values, "utc": d["target_time_utc"].values, "lead": d["lead_hours"].values,
                       "p10": (d["p10"] * k).values, "p50": (d["p50"] * k).values, "p90": (d["p90"] * k).values,
                       "wind": d["wind_speed_100m"].values})
    cur_name = f"P50 выпуска {res.issue_label}"
    prev_name = None
    p = res.previous
    if p is not None and not p.empty and series in set(p["turbine"]):
        pr = p[p["turbine"] == series][["target_time_utc", "p50"]].rename(columns={"p50": "prev", "target_time_utc": "utc"})
        df = df.merge(pr, on="utc", how="left")
        df["prev"] = df["prev"] * k
        prev_name = f"P50 выпуска {res.previous_issue}" if df["prev"].notna().any() else None
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
    names = [cur_name] + ([prev_name] if prev_name else []) + ([fact_name] if fact_name else []) + ["Ветер 100 м, м/с"]
    colors = [pal["p50"]] + ([pal["prev"]] if prev_name else []) + ([pal["fact"]] if fact_name else []) + [pal["wind"]]
    dashes = [[1, 0]] + ([[5, 4]] if prev_name else []) + ([[1, 0]] if fact_name else []) + [[1, 0]]
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
    layers.append(alt.Chart(long).mark_line(strokeWidth=1.8).encode(
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

    wind_df = pd.DataFrame({"t": df["t"], "s": "Ветер 100 м, м/с", "v": df["wind"]})
    wind = alt.Chart(wind_df).mark_line(strokeWidth=1.2, opacity=0.85).encode(
        x=x, y=alt.Y("v:Q", title="Ветер 100 м, м/с", axis=alt.Axis(orient="right", grid=False)), color=color, strokeDash=dash)
    return alt.layer(alt.layer(*layers), wind).resolve_scale(y="independent").properties(height=320)


def main_chart(res: core.ForecastResult, view: pd.DataFrame) -> str:
    site = res.site
    c1, c2 = st.columns([3, 2])
    sel = c1.radio("Ряд", res.series, format_func=series_label, horizontal=True, key=f"series_{site.key}")
    units = (["mw"] if site.capacity_mw(sel) else []) + ["frac"]
    unit = c2.radio("Единицы", units, format_func=UNIT_LABELS.get, horizontal=True, key=f"unit_{site.key}")
    d = view[view["turbine"] == sel]
    k = site.capacity_mw(sel) if unit == "mw" else 1.0
    fact = core.actual_for(view, sel, scada_actual()) if not site.transfer else None
    st.altair_chart(fan_chart(res, d, sel, k, unit, palette(), fact), **_stretch(st.altair_chart))
    return unit


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
    st.dataframe(table.rename(columns={"Сутки (местные)": "Сутки"}), hide_index=True, **_stretch(st.dataframe))
    with st.expander(f"Почасовой прогноз, {horizon} ч"):
        st.dataframe(core.hourly_table(view, res.series, series_label, scale, "МВт" if unit == "mw" else "доля ном."),
                     hide_index=True, height=420, **_stretch(st.dataframe))
    stem = f"{site.key}_{res.issue_label}" + ("" if res.params.mode == core.MODE_LIVE else f"T{res.params.issue_hour:02d}")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Экспорт CSV", data=core.forecast_csv(view), file_name=f"forecast_{stem}_{horizon}h.csv",
                       mime="text/csv", help="Контракт FORECAST_COLUMNS", **_stretch(st.download_button))
    c2.download_button("Экспорт MD", data=res.report_md.encode("utf-8"), file_name=f"report_{stem}.md",
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
                 column_config={"Шаг": cc.NumberColumn(width="small"), "Время, с": cc.NumberColumn(format="%.2f", width="small"),
                                "Итог": cc.TextColumn(width="large")})
    with st.expander("Отчёт агента"):
        if res.report_md:
            st.markdown(res.report_md)
    if res.core_log:
        with st.expander(f"Журнал ядра ({len(res.core_log)})"):
            for lvl, msg in res.core_log:
                st.text(f"[{lvl}] {msg}")


def tab_site(res: core.ForecastResult) -> None:
    site = res.site
    pts = pd.DataFrame([(series_label(k), la, lo) for k, la, lo in site.points], columns=["Турбина", "lat", "lon"])
    c1, c2 = st.columns([3, 2])
    with c1:
        st.map(pts, latitude="lat", longitude="lon",
               **_kw(st.map, zoom=13 if not site.transfer else 7, size=40 if not site.transfer else 3000,
                     color="#8d96a3", height=320))
    with c2:
        rows = [("Название", site.name), ("Ключ", site.key), ("Турбин", str(site.n_turbines)),
                ("Номинал турбины", f"{site.rated_mw:.1f} МВт" if site.rated_mw else "—"),
                ("Мощность площадки", f"{site.n_turbines * site.rated_mw:.1f} МВт" if site.rated_mw else "—"),
                ("История SCADA", "нет, перенос модели" if site.transfer else "есть, модель обучена")]
        rows += [(f"{series_label(k)}, °", f"{la:.5f}, {lo:.5f}") for k, la, lo in site.points]
        st.dataframe(pd.DataFrame(rows, columns=["Параметр", "Значение"]), hide_index=True, **_stretch(st.dataframe))


def tab_fact(res: core.ForecastResult) -> None:
    st.markdown("**Факт SCADA**")
    act = {} if res.site.transfer else scada_actual()
    up = st.file_uploader("Файл факта (формат data/raw/turbine_*.csv)", type=["csv"], accept_multiple_files=True,
                          key="fact_upload")
    if up:
        try:
            act = core.load_uploaded_actual(up)
            if len(act) > 1:
                act[core.FARM] = pd.concat(act.values(), axis=1).mean(axis=1)
        except Exception as e:  # noqa: BLE001
            st.error(f"Файл не прочитан: {type(e).__name__}: {e}")
            return
    if not act:
        st.markdown('<div class="wa-alert wa-alert-info">Факта для площадки нет</div>', unsafe_allow_html=True)
        return
    ev = core.accuracy(res.forecast, act)
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
                 "coverage_p10_p90_pct": "Покрытие P10–P90, %", "mean_actual": "Факт ср.", "mean_p50": "P50 ср."})
    st.dataframe(out.round(3), hide_index=True, **_stretch(st.dataframe))
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


def tab_3d(res: core.ForecastResult) -> None:
    if res.site.custom_requested or res.site.transfer:
        st.markdown('<div class="wa-banner">Сцена построена для ВЭС «Нурлы»</div>', unsafe_allow_html=True)
    stamp = tuple(int(f.stat().st_mtime) if f.exists() else 0 for f in (VIZ_DIR / "index.html", VIZ_DIR / "data" / "viz_data.js"))
    issue = res.params.issue_date.isoformat() if res.params.mode == core.MODE_REPLAY else str(res.issue_label)
    htm = viz_html(issue, stamp)
    if htm is None:
        st.markdown('<div class="wa-alert wa-alert-info">Данных сцены нет: make viz</div>', unsafe_allow_html=True)
        return
    if components is not None and hasattr(components, "html"):
        components.html(htm, height=760, scrolling=False)
    elif hasattr(st, "iframe"):
        st.iframe(htm, height=760)


# ---------------------------------------------------------------- страница
def show_error(err) -> None:
    st.error(err[0])
    with st.expander("Подробности"):
        st.code(err[1])


def main() -> None:
    params, run = sidebar()
    if run or ("result" not in st.session_state and "error" not in st.session_state and not params.use_llm):
        run_and_store(params)          # первый заход — сразу расчёт по параметрам по умолчанию (из кэша, < 1 с)

    res: core.ForecastResult | None = st.session_state.get("result")
    st.title(f"WindAgent · {title_for(params, res)}")
    custom_banner = params.site == core.CUSTOM_KEY or (res is not None and res.site.custom_requested)
    err = st.session_state.get("error")
    if res is None:
        if custom_banner:
            st.markdown(f'<div class="wa-banner">{esc(core.TRANSFER_WARNING)}</div>', unsafe_allow_html=True)
        if err:
            show_error(err)
        return

    view = res.view(params.horizon)
    header(res, view, params.horizon, stale=res.params.run_key() != params.run_key())
    if custom_banner:
        st.markdown(f'<div class="wa-banner">{esc(core.TRANSFER_WARNING)}</div>', unsafe_allow_html=True)
    if err:
        show_error(err)
    toast = st.session_state.pop("toast", None)
    if toast and _toast:
        _toast(toast)

    kpis(res, view, params.horizon)
    unit = main_chart(res, view)

    rows = core.alert_rows(res, view, custom_banner)
    n_att = sum(r["level"] in ("warning", "error") for r in rows)
    tabs = st.tabs([f"Предупреждения ({n_att})", "Таблицы и выгрузка", "Агент", "Площадка", "Факт и точность", "3D-сцена"])
    with tabs[0]:
        tab_alerts(rows)
    with tabs[1]:
        tab_tables(res, view, params.horizon, unit)
    with tabs[2]:
        tab_agent(res)
    with tabs[3]:
        tab_site(res)
    with tabs[4]:
        tab_fact(res)
    with tabs[5]:
        tab_3d(res)


main()
