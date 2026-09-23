"""Оркестратор агента: полный цикл «погода → подготовка → модель → анализ → решение/пересчёт → отчёт».

Один и тот же набор инструментов (tools.py) вызывает либо LLM через tool calling (llm.py), либо
детерминированные правила — в том же порядке. Если LLM недоступна или не довела цикл до конца,
оставшиеся шаги выполняются по правилам с сохранением уже сделанного.

replay — ежедневные выпуски тестового периода «как в прошлом» (архивные прогнозы погоды);
live   — оперативный прогноз по последнему запуску Open-Meteo.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from . import llm, tools

log = logging.getLogger(__name__)

TOOLS = ("fetch_weather", "prepare_data", "run_model", "analyze_forecast", "recalculate", "write_report")
PIPELINE = ("fetch_weather", "prepare_data", "run_model", "analyze_forecast")
_STATE_KEY = {"fetch_weather": "weather", "prepare_data": "prepared", "run_model": "forecast", "analyze_forecast": "analysis"}
RECALC_CODES = {"input_quality", "range_violation", "quantile_order", "incomplete", "revision"}

FORECASTS_DIR = config.OUTPUTS_DIR / "forecasts"
REPORTS_DIR = config.OUTPUTS_DIR / "reports"
LIVE_DIR = config.OUTPUTS_DIR / "live"
ADHOC_DIR = config.OUTPUTS_DIR / "adhoc"      # прогнозы для других площадок / нестандартного часа выпуска
AGENT_LOG = config.OUTPUTS_DIR / "agent_log.jsonl"
REPLAY_SUMMARY = config.OUTPUTS_DIR / "replay_summary.csv"
LATEST_BY_TARGET = FORECASTS_DIR / "latest_by_target.csv"


@dataclass
class LLMState:
    """Доступность LLM на весь прогон: после двух подряд сбоев API отключаем её до конца прогона."""
    settings: dict | None = None
    client: object = None
    failures: int = 0

    @property
    def enabled(self) -> bool:
        return self.settings is not None and self.failures < 2


class AgentSession:
    """Состояние одного выпуска прогноза и журнал шагов агента."""

    def __init__(self, wx, issue_date: str | None = None, live: bool = False, previous: pd.DataFrame | None = None,
                 now: pd.Timestamp | None = None, out_name: str | None = None, site: config.Site | None = None,
                 issue_hour: int | None = None):
        self.wx, self.issue_date, self.live, self.previous = wx, issue_date, live, previous
        self.now = now if now is not None else (pd.Timestamp.now(tz="UTC") if live else None)
        self.out_name = out_name
        self.site = site or config.NURLY
        self.issue_hour = config.ISSUE_HOUR_LOCAL if issue_hour is None else int(issue_hour)
        # протокол ТЗ (файлы outputs/forecasts и outputs/reports) — только ВЭС «Нурлы» с выпуском в 23:00;
        # другие площадки и часы выпуска пишутся в outputs/adhoc/<площадка>/, чтобы не смешивать с тестовым прогоном
        self.protocol = self.site.has_history and self.issue_hour == config.ISSUE_HOUR_LOCAL
        self.state: dict = {}
        self.trace: list[dict] = []
        self.recalc: dict | None = None
        self.final: dict | None = None
        self.calls0 = wx.calls
        self.llm_model: str | None = None

    # ------------------------------------------------------------ журнал
    def _record(self, tool: str, status: str, t0: float, summary: str, by_llm: bool) -> None:
        self.trace.append({"step": len(self.trace) + 1, "tool": tool, "status": status,
                           "duration_s": round(time.perf_counter() - t0, 3), "summary": summary, "llm_used": by_llm})
        log.debug("  %s [%s] %s", tool, status, summary)

    def log_llm_turn(self, step: int, duration: float, calls: list[str], tokens: int | None) -> None:
        self.trace.append({"step": len(self.trace) + 1, "tool": "llm", "status": "ok", "duration_s": round(duration, 3),
                           "summary": f"ход {step}: " + (", ".join(calls) if calls else "текстовый ответ")
                                      + (f"; токенов {tokens}" if tokens else ""), "llm_used": True})

    # ------------------------------------------------------------ вызовы
    def call(self, tool: str, by_llm: bool = False, **kw) -> dict:
        """Выполнить инструмент с таймингом и записью в журнал; ошибки записываются и пробрасываются."""
        if tool not in TOOLS:
            raise ValueError(f"неизвестный инструмент {tool}")
        t0 = time.perf_counter()
        try:
            summary, result = getattr(self, "_" + tool)(by_llm=by_llm, **kw)
        except Exception as e:
            self._record(tool, "error", t0, f"ошибка: {e}", by_llm)
            raise
        self._record(tool, "ok", t0, summary, by_llm)
        return result

    def dispatch(self, tool: str, args: dict) -> dict:
        """Вызов от LLM: ошибки не роняют цикл, а возвращаются модели как {'error': ...}."""
        if tool not in TOOLS:
            return {"error": f"неизвестный инструмент {tool}; доступны: {', '.join(TOOLS)}"}
        allowed = {"recalculate": {"reason"}, "write_report": {"decision", "reasoning", "narrative_ru"}}.get(tool, set())
        try:
            return self.call(tool, by_llm=True, **{k: v for k, v in args.items() if k in allowed})
        except Exception as e:  # noqa: BLE001 — модель должна увидеть ошибку и принять решение
            return {"error": str(e)}

    def _ensure(self, tool: str, by_llm: bool) -> None:
        """Выполнить пропущенные предыдущие шаги (если LLM вызвала инструменты не по порядку)."""
        for t in PIPELINE[:PIPELINE.index(tool)]:
            if _STATE_KEY[t] not in self.state:
                self.call(t, by_llm=by_llm)

    # ------------------------------------------------------------ инструменты
    def _fetch_weather(self, by_llm: bool = False):
        w = tools.fetch_weather(self.wx, self.issue_date, live=self.live, now=self.now, site=self.site, issue_hour=self.issue_hour)
        for k in ("prepared", "forecast", "analysis"):
            self.state.pop(k, None)
        self.state["weather"] = w
        self.issue_date = w["meta"]["issue_date"]
        m = w["meta"]
        return tools.weather_summary(m), {k: m[k] for k in ("source", "mode", "issue_date", "issue_time_utc", "target_start_local",
                                                            "target_end_local", "n_hours", "n_nan", "ensemble_members", "input_hash")}

    def _prepare_data(self, by_llm: bool = False):
        self._ensure("prepare_data", by_llm)
        p = tools.prepare_data(self.state["weather"])
        self.state["prepared"] = p
        for k in ("forecast", "analysis"):
            self.state.pop(k, None)
        n_feat = {t: list(X.shape) for t, X in p["features"].items()}
        summary = f"признаки {n_feat}; замечаний {len(p['notes'])}" + (f": {'; '.join(p['notes'])}" if p["notes"] else "")
        return summary, {"features_shape": n_feat, "notes": p["notes"], "info": p["info"], "ok": not p["notes"]}

    def _run_model(self, by_llm: bool = False):
        self._ensure("run_model", by_llm)
        fc = tools.run_model(self.state["prepared"])
        self.state["forecast"] = fc
        self.state.pop("analysis", None)
        farm = fc[fc["turbine"] == "farm"]
        brief = {"rows": len(fc), "model_version": fc["model_version"].iloc[0],
                 "farm_p50_mean": round(float(farm["p50"].mean()), 3), "farm_p50_sum_48h": round(float(farm["p50"].sum()), 2)}
        return tools.forecast_summary(fc), brief

    def _analyze_forecast(self, by_llm: bool = False):
        self._ensure("analyze_forecast", by_llm)
        p = self.state["prepared"]
        history = tools.load_history() if self.site.has_history else None
        a = tools.analyze_forecast(self.state["forecast"], self.previous, history, notes=p["notes"], info=p["info"], site=self.site)
        self.state["analysis"] = a
        f = a["metrics"]["farm"]
        rev = a["revision"]["farm"]["mae"] if a["revision"] else None
        summary = (f"статус {a['status']}; парк {f['energy_48h']:.1f} ч.н. за 48 ч, P50 ср. {f['mean_p50']:.2f}, "
                   f"рампа {f['max_ramp']:.2f}, ревизия MAE {rev if rev is not None else '—'}; флаги: "
                   + (", ".join(x["code"] for x in a["flags"]) or "нет"))
        return summary, tools.analysis_brief(a)

    def _recalculate(self, by_llm: bool = False, reason: str = ""):
        """Повторный запрос входа и пересчёт: в replay архив не меняется (подтверждаем), в live вход мог обновиться."""
        if "forecast" not in self.state:
            self._ensure("analyze_forecast", by_llm)
        old_w, old_fc = self.state["weather"], self.state["forecast"]
        new_w = tools.fetch_weather(self.wx, self.issue_date, live=self.live, now=self.now, site=self.site, issue_hour=self.issue_hour)
        new_p = tools.prepare_data(new_w)
        new_fc = tools.run_model(new_p)
        changed = new_w["meta"]["input_hash"] != old_w["meta"]["input_hash"]
        diff = float(np.abs(new_fc["p50"].to_numpy() - old_fc["p50"].to_numpy()).max()) if len(new_fc) == len(old_fc) else float("nan")
        if changed:
            self.state.update(weather=new_w, prepared=new_p, forecast=new_fc)
            self.state.pop("analysis", None)
            self._analyze_forecast(by_llm)
        self.recalc = {"reason": tools.clean_text(reason, 300), "input_changed": changed, "old_hash": old_w["meta"]["input_hash"],
                       "new_hash": new_w["meta"]["input_hash"], "max_abs_diff": round(diff, 4)}
        verdict = ("вход обновился — прогноз пересчитан и переанализирован" if changed else
                   "вход не изменился (архивный прогноз неизменен) — прогноз подтверждён" if not self.live else
                   "вход не изменился (новый запуск Open-Meteo ещё не вышел) — прогноз подтверждён")
        result = dict(self.recalc, verdict=verdict)
        if changed:
            result["analysis"] = tools.analysis_brief(self.state["analysis"])
        return f"{verdict}; хэш {self.recalc['old_hash']} → {self.recalc['new_hash']}, max |ΔP50| {diff:.4f}", result

    def _fact_check(self, narrative: str, analysis: dict, forecast: pd.DataFrame, by_llm: bool) -> tuple[str, dict]:
        """Проверка фактов нарратива (шаг verify_narrative в трассе и журнале).

        Текст LLM: 1 неподтверждённое число — текст остаётся с пометкой; 2 и более — заменяется шаблоном.
        Шаблонный текст строится из тех же чисел анализа и проверку проходит всегда (см. tests/test_agent.py).
        """
        t0 = time.perf_counter()
        fact = tools.verify_narrative(narrative, analysis, forecast)
        bad = fact["unverified"]
        fact.update(note=None, action="accepted", rejected_narrative=None, source="llm" if by_llm else "template")
        if by_llm and len(bad) == 1:
            fact.update(note=f"⚠ Не подтверждено анализом: {bad[0]}", action="annotated")
            status, summary = "warning", f"проверено чисел {fact['checked']}, не подтверждено 1 ({bad[0]}) — текст LLM оставлен с пометкой"
        elif by_llm and len(bad) >= 2:
            fact.update(note=f"Комментарий LLM отклонён проверкой фактов (не подтверждено анализом: {', '.join(bad)}), "
                             "использован шаблон", action="replaced", rejected_narrative=narrative, source="template")
            narrative = tools.template_narrative(analysis)
            status, summary = "warning", (f"проверено чисел {fact['checked']}, не подтверждено {len(bad)} ({', '.join(bad)}) — "
                                          "нарратив LLM заменён шаблоном")
            log.warning("%s: %s", self.issue_date, fact["note"])
        elif bad:   # шаблон не прошёл проверку — сигнал об ошибке в самой проверке
            status, summary = "warning", f"шаблон: не подтверждено {len(bad)} ({', '.join(bad)})"
            log.warning("%s: шаблонный нарратив не прошёл проверку фактов: %s", self.issue_date, bad)
        else:
            status, summary = "ok", f"проверено чисел {fact['checked']}, все подтверждены анализом"
        self._record("verify_narrative", status, t0, summary, by_llm)
        return narrative, fact

    def _write_report(self, by_llm: bool = False, decision: str = "", reasoning: str = "", narrative_ru: str = ""):
        clean, err = llm.validate_final({"decision": decision, "reasoning": reasoning, "narrative_ru": narrative_ru})
        if clean is None:
            raise ValueError(f"ответ не прошёл валидацию: {err}")
        if "analysis" not in self.state:
            self._ensure("analyze_forecast", by_llm)
            self.call("analyze_forecast", by_llm=by_llm)
        rules_decision, rules_reason = rule_decision(self.state["analysis"])
        # guardrail: обязательный пересчёт по правилам выполняется всегда, даже если LLM решила его пропустить;
        # решение LLM сохраняется рядом с решением правил, расхождение видно в отчёте
        if self.recalc is None and (clean["decision"] == "recalculate" or rules_decision == "recalculate"):
            why = clean["reasoning"] if clean["decision"] == "recalculate" else f"по правилам: {rules_reason}"
            self.call("recalculate", by_llm=by_llm, reason=why)
            rules_decision, rules_reason = rule_decision(self.state["analysis"])
        a, fc, meta = self.state["analysis"], self.state["forecast"], self.state["weather"]["meta"]
        narrative, fact = self._fact_check(clean["narrative_ru"], a, fc, by_llm)
        dec = {"decision": clean["decision"], "reasoning": clean["reasoning"], "llm_used": by_llm, "model": self.llm_model or "",
               "rules_decision": rules_decision, "recalc": self.recalc, "fact_check": fact}
        if self.live:
            out_dir = LIVE_DIR if self.site.has_history else LIVE_DIR / self.site.key
            csv_path = tools.save_forecast(self.issue_date, fc, out_dir=out_dir, name=self.out_name)
            md_path = tools.write_report(self.issue_date, fc, a, narrative, meta=meta, decision=dec,
                                         trace=self.trace, out_dir=out_dir, name=self.out_name)
        elif self.protocol:
            csv_path = tools.save_forecast(self.issue_date, fc)
            md_path = tools.write_report(self.issue_date, fc, a, narrative, meta=meta, decision=dec, trace=self.trace)
        else:   # другая площадка или нестандартный час выпуска — отдельная папка, протокольные файлы не трогаем
            out_dir = ADHOC_DIR / self.site.key
            name = self.out_name or f"{self.issue_date}T{self.issue_hour:02d}"
            csv_path = tools.save_forecast(self.issue_date, fc, out_dir=out_dir, name=name)
            md_path = tools.write_report(self.issue_date, fc, a, narrative, meta=meta, decision=dec,
                                         trace=self.trace, out_dir=out_dir, name=name)
        self.final = dict(dec, narrative=narrative, csv=str(csv_path), report=str(md_path))
        rel = lambda p: str(Path(p).relative_to(config.ROOT)) if str(p).startswith(str(config.ROOT)) else str(p)  # noqa: E731
        return (f"решение {clean['decision']} ({'LLM' if by_llm else 'правила'}); {rel(md_path)}, {rel(csv_path)}",
                {"status": "ok", "report": rel(md_path), "forecast_csv": rel(csv_path)})


# ---------------------------------------------------------------- правила
def rule_decision(analysis: dict) -> tuple[str, str]:
    """Детерминированное решение по флагам анализа: recalculate | flag | accept с обоснованием."""
    warn = [f for f in analysis["flags"] if f["level"] == "warning"]
    codes = {f["code"] for f in warn}
    if codes & RECALC_CODES:
        why = "; ".join(f["message"] for f in warn if f["code"] in RECALC_CODES)
        return "recalculate", f"Нужна перепроверка входа повторным запросом: {why}."
    if warn:
        return "flag", "Прогноз корректен, но требует внимания диспетчера: " + "; ".join(f["message"] for f in warn) + "."
    return "accept", "Все проверки пройдены, флагов уровня warning нет — прогноз принимается."


def _complete_by_rules(session: AgentSession) -> None:
    """Довести цикл до конца по правилам, переиспользуя уже выполненные шаги."""
    for t in PIPELINE:
        if _STATE_KEY[t] not in session.state:
            session.call(t)
    decision, reasoning = rule_decision(session.state["analysis"])
    if decision == "recalculate" and session.recalc is None:
        session.call("recalculate", reason=reasoning)
    narrative = tools.template_narrative(session.state["analysis"])
    session.call("write_report", decision=decision, reasoning=reasoning, narrative_ru=narrative)


def _llm_context(session: AgentSession) -> str:
    if session.live:
        head = (f"Режим: live (оперативный прогноз). Текущее время UTC: {session.now:%Y-%m-%d %H:%M}. "
                "Прогноз на 48 часов вперёд по последнему запуску Open-Meteo.")
    else:
        head = (f"Режим: replay (ретроспективный прогон). Дата выпуска: {session.issue_date}, выпуск в {session.issue_hour:02d}:00 "
                "местного времени (UTC+5), прогноз на следующие 48 часов по архивным прогнозам погоды, доступным на момент выпуска.")
    s = session.site
    site_txt = (f"Площадка: {s.name}; турбины: {', '.join(s.turbines)}"
                + (f"; {s.n_turbines} × {s.rated_mw} МВт" if s.rated_mw and s.n_turbines else "")
                + (". ВНИМАНИЕ: истории SCADA у площадки нет, прогноз — перенос модели ВЭС «Нурлы», об этом нужно сказать в отчёте."
                   if s.transfer else "."))
    prev = (f"Предыдущий выпуск для анализа ревизии: {session.previous['issue_date'].iloc[0]}."
            if session.previous is not None else "Предыдущего выпуска для сравнения нет.")
    return f"{head} {site_txt} {prev} Выполни полный цикл агента и заверши его вызовом write_report."


def run_cycle(wx, llm_state: LLMState, issue_date: str | None = None, live: bool = False,
              previous: pd.DataFrame | None = None, out_name: str | None = None, now: pd.Timestamp | None = None,
              site: config.Site | None = None, issue_hour: int | None = None) -> AgentSession:
    """Один выпуск прогноза: LLM-цикл (если доступна) с доведением по правилам при сбое/исчерпании шагов.

    site — площадка (по умолчанию ВЭС «Нурлы»; для произвольных координат — config.custom_site(...)),
    issue_hour — час выпуска местного времени (по умолчанию 23, протокол ТЗ).
    """
    session = AgentSession(wx, issue_date=issue_date, live=live, previous=previous, now=now, out_name=out_name,
                           site=site, issue_hour=issue_hour)
    if llm_state.enabled:
        session.llm_model = llm_state.settings["model"]
        try:
            if llm_state.client is None:
                llm_state.client = llm.make_client()
            llm.run_llm_cycle(session, llm_state.client, llm_state.settings["model"], _llm_context(session),
                              reasoning_effort=llm_state.settings.get("reasoning_effort"))
            llm_state.failures = 0
        except Exception as e:  # noqa: BLE001 — любой сбой API не должен ронять прогон
            llm_state.failures += 1
            session.trace.append({"step": len(session.trace) + 1, "tool": "llm", "status": "error", "duration_s": 0.0,
                                  "summary": f"сбой LLM ({type(e).__name__}) — дальше по правилам", "llm_used": True})
            log.warning("LLM недоступна (%s: %s) — продолжаю по детерминированным правилам%s", type(e).__name__, str(e)[:200],
                        "; LLM отключена до конца прогона" if not llm_state.enabled else "")
    if session.final is None:
        _complete_by_rules(session)
    return session


# ---------------------------------------------------------------- файлы прогона
def _read_forecast(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _upsert_log(entries: list[dict], replace_dates: set[str] | None = None, mode: str = "replay") -> None:
    """Дописать шаги в agent_log.jsonl; при повторном replay тех же дат старые записи этих дат заменяются."""
    AGENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    if replace_dates and AGENT_LOG.exists():
        for line in AGENT_LOG.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (rec.get("mode") == mode and rec.get("issue_date") in replace_dates):
                kept.append(line)
        AGENT_LOG.write_text("".join(x + "\n" for x in kept), encoding="utf-8")
    with AGENT_LOG.open("a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")


def _log_entries(session: AgentSession, mode: str, run_ts: str) -> list[dict]:
    return [{"ts": run_ts, "mode": mode, "issue_date": session.issue_date, "step": t["step"], "tool": t["tool"],
             "status": t["status"], "duration_s": t["duration_s"], "summary": t["summary"], "llm_used": t["llm_used"]}
            for t in session.trace]


def _summary_row(session: AgentSession) -> dict:
    a = session.state["analysis"]
    f = a["metrics"]["farm"]
    return {"issue_date": session.issue_date, "farm_energy_48h": f["energy_48h"], "farm_energy_day1": f["energy_day1"],
            "farm_energy_day2": f["energy_day2"], "mean_p50": f["mean_p50"],
            "revision_mae": a["revision"]["farm"]["mae"] if a["revision"] else np.nan,
            "flags": ";".join(x["code"] for x in a["flags"]), "status": a["status"],
            "decision": session.final["decision"], "llm_used": bool(session.final["llm_used"]),
            "api_calls": int(session.wx.calls - session.calls0)}


def build_latest_by_target(forecast_dir: Path = FORECASTS_DIR) -> pd.DataFrame:
    """Для каждого целевого часа — самый свежий прогноз (lead_day=1 перекрывает lead_day=2 вчерашнего выпуска),
    плюс previous_p50 (прогноз предыдущего выпуска на тот же час) и revision = p50 − previous_p50."""
    files = sorted(p for p in forecast_dir.glob("????-??-??.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["_t"] = pd.to_datetime(df["target_time_utc"], utc=True)
    df = df.sort_values(["turbine", "_t", "issue_date"], kind="stable")
    df["previous_p50"] = df.groupby(["turbine", "_t"])["p50"].shift(1)
    latest = df.groupby(["turbine", "_t"], sort=False).tail(1).copy()
    latest["revision"] = (latest["p50"] - latest["previous_p50"]).round(4)
    latest["_o"] = latest["turbine"].map({s: i for i, s in enumerate(tools.SERIES)})
    latest = latest.sort_values(["_o", "_t"]).reset_index(drop=True)
    return latest[tools.FORECAST_COLUMNS + ["previous_p50", "revision"]]


# ---------------------------------------------------------------- режимы
def _init_llm(use_llm: bool) -> LLMState:
    if not use_llm:
        log.info("Режим без LLM (--no-llm): шаги агента выполняются по детерминированным правилам")
        return LLMState()
    settings = llm.llm_settings()
    if settings is None:
        log.warning("OPENAI_API_KEY не задан — агент работает в детерминированном режиме (те же шаги, отчёт по шаблону)")
        return LLMState()
    log.info("LLM: %s (tool calling, не более %d шагов на выпуск)", settings["model"], llm.MAX_STEPS)
    return LLMState(settings=settings)


def run_replay(wx, start: str = config.TEST_ISSUE_START, end: str = config.TEST_ISSUE_END, use_llm: bool = True,
               site: config.Site | None = None, issue_hour: int | None = None) -> pd.DataFrame:
    """Ежедневные выпуски start..end (местные даты): полный цикл агента на каждую дату, csv + md + журнал + сводка.

    Для ВЭС «Нурлы» с выпуском в 23:00 (протокол ТЗ) файлы идут в outputs/forecasts и outputs/reports;
    другая площадка или час выпуска — в outputs/adhoc/<площадка>/ (журнал и сводка протокола не трогаются).
    """
    t_run = time.perf_counter()
    run_ts = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    site = site or config.NURLY
    protocol = site.has_history and (issue_hour is None or int(issue_hour) == config.ISSUE_HOUR_LOCAL)
    llm_state = _init_llm(use_llm)
    if protocol:
        try:
            wx.prefetch()
        except Exception as e:  # noqa: BLE001 — без префетча попробуем по датам
            log.warning("prefetch не выполнен (%s) — продолжаю по датам", e)
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]
    log.info("Replay: %d выпусков %s..%s, площадка %s, выпуск в %02d:00%s", len(dates), dates[0], dates[-1], site.name,
             config.ISSUE_HOUR_LOCAL if issue_hour is None else int(issue_hour), "" if protocol else " (вне протокола → outputs/adhoc)")
    rows, entries, done, prev_mem = [], [], set(), {}
    for d in dates:
        prev_date = (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        previous = prev_mem.get(prev_date)
        if previous is None and protocol:
            previous = _read_forecast(FORECASTS_DIR / f"{prev_date}.csv")
        try:
            s = run_cycle(wx, llm_state, issue_date=d, previous=previous, site=site, issue_hour=issue_hour)
        except Exception as e:  # noqa: BLE001 — одна дата не должна ронять весь прогон
            log.error("%s | ошибка цикла: %s", d, e)
            rows.append({"issue_date": d, "status": "error", "flags": f"error: {e}"[:200], "llm_used": False})
            continue
        prev_mem[d] = s.state["forecast"]
        row = _summary_row(s)
        rows.append(row)
        entries += _log_entries(s, "replay", run_ts)
        done.add(d)
        rev = f"{row['revision_mae']:.3f}" if pd.notna(row["revision_mae"]) else "—"
        log.info("%s | парк %5.1f ч.н. за 48 ч (D+1 %4.1f, D+2 %4.1f) | P50 ср. %.2f | ревизия MAE %s | %-7s | %-11s | LLM %s | %s",
                 d, row["farm_energy_48h"], row["farm_energy_day1"], row["farm_energy_day2"], row["mean_p50"], rev,
                 row["status"], row["decision"], "да" if row["llm_used"] else "нет", row["flags"] or "без флагов")

    summary = pd.DataFrame(rows)
    cols = ["issue_date", "farm_energy_48h", "farm_energy_day1", "farm_energy_day2", "mean_p50", "revision_mae",
            "flags", "status", "llm_used", "api_calls", "decision"]
    summary = summary.reindex(columns=cols)
    if not protocol:   # вне протокола: журнал с пометкой adhoc, сводка и latest_by_target протокола не меняются
        _upsert_log(entries, mode="adhoc")
        log.info("Replay (adhoc) завершён за %.1f с: выпусков %d, файлы в %s", time.perf_counter() - t_run, len(summary),
                 (ADHOC_DIR / site.key).relative_to(config.ROOT))
        return summary
    _upsert_log(entries, replace_dates=done, mode="replay")
    if REPLAY_SUMMARY.exists():
        old = pd.read_csv(REPLAY_SUMMARY)
        old = old[~old["issue_date"].isin(summary["issue_date"])]
        summary_all = pd.concat([old.reindex(columns=cols), summary], ignore_index=True).sort_values("issue_date")
    else:
        summary_all = summary
    summary_all.to_csv(REPLAY_SUMMARY, index=False)
    latest = build_latest_by_target()
    if not latest.empty:
        latest.to_csv(LATEST_BY_TARGET, index=False)
    ok = summary[summary["status"] != "error"]
    n_rev = int((ok["flags"].fillna("").str.contains("revision")).sum())
    log.info("Replay завершён за %.1f с: выпусков %d (ошибок %d), средняя энергия парка %.1f ч.н./48 ч, "
             "существенных ревизий %d, LLM-выпусков %d, запросов к API %d → %s, %s",
             time.perf_counter() - t_run, len(ok), len(summary) - len(ok), ok["farm_energy_48h"].mean() if len(ok) else float("nan"),
             n_rev, int(ok["llm_used"].sum()) if len(ok) else 0, int(ok["api_calls"].sum()) if len(ok) else 0,
             _rel(REPLAY_SUMMARY), _rel(LATEST_BY_TARGET))
    return summary


def _rel(p: Path | str) -> str:
    """Путь относительно корня проекта (или как есть, если он снаружи — например, в тестах)."""
    p = Path(p)
    try:
        return str(p.relative_to(config.ROOT))
    except ValueError:
        return str(p)


def run_live(wx, use_llm: bool = True, site: config.Site | None = None) -> AgentSession | None:
    """Оперативный прогноз на 48 ч от текущего момента: outputs/live/<UTC timestamp>.csv и .md
    (для другой площадки — outputs/live/<площадка>/)."""
    t_run = time.perf_counter()
    if wx.offline:
        log.error("live требует доступа к Open-Meteo Forecast API: оперативный прогноз не кэшируется, запустите без --offline "
                  "(для офлайн-проверки используйте replay)")
        return None
    site = site or config.NURLY
    llm_state = _init_llm(use_llm)
    now = pd.Timestamp.now(tz="UTC")
    name = (now.floor("h")).strftime("%Y%m%dT%H%MZ")
    live_dir = LIVE_DIR if site.has_history else LIVE_DIR / site.key
    live_dir.mkdir(parents=True, exist_ok=True)
    prev_files = sorted(p for p in live_dir.glob("*Z.csv") if p.stem != name)
    previous = pd.read_csv(prev_files[-1]) if prev_files else None
    try:
        s = run_cycle(wx, llm_state, live=True, previous=previous, out_name=name, now=now, site=site)
    except Exception as e:  # noqa: BLE001
        log.error("live: ошибка цикла агента: %s", e)
        return None
    _upsert_log(_log_entries(s, "live", now.strftime("%Y-%m-%dT%H:%M:%SZ")), mode="live")
    row = _summary_row(s)
    log.info("Live %s | парк %.1f ч.н. за 48 ч (первые сутки %.1f, вторые %.1f) | P50 ср. %.2f | %s | %s | LLM %s | %.1f с → %s",
             name, row["farm_energy_48h"], row["farm_energy_day1"], row["farm_energy_day2"], row["mean_p50"], row["status"],
             row["decision"], "да" if row["llm_used"] else "нет", time.perf_counter() - t_run, _rel(s.final["report"]))
    return s
