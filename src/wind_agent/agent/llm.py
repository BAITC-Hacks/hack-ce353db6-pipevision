"""LLM-слой агента: OpenAI tool calling поверх инструментов tools.py с честным детерминированным fallback.

Ключ берётся из env `OPENAI_API_KEY` (cli делает load_dotenv(.env)), модель — из `OPENAI_MODEL`
(по умолчанию gpt-5-mini), `OPENAI_BASE_URL` SDK читает сам. Нет ключа или вызов упал — агент продолжает
по детерминированным правилам, прогон не падает. Всё, что возвращает LLM, валидируется.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

from .tools import clean_text

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5-mini"
MAX_STEPS = 8          # максимум обращений к LLM за один выпуск прогноза
TIMEOUT_S = 60
DECISIONS = ("accept", "recalculate", "flag")

SYSTEM_PROMPT = """Ты — агент-прогнозист ветроэлектростанции (ВЭС) в Казахстане; площадка и её турбины указаны в контексте
(по умолчанию ВЭС «Нурлы», турбины t1 и t2, на истории которой обучена модель; для другой площадки прогноз — перенос модели).
Мощность нормирована: 1.0 = номинал турбины; «часы номинала» = сумма почасовой нормированной мощности.
Задача: выпустить почасовой прогноз выработки на 48 часов (P10/P50/P90 для каждой турбины и парка) и отчёт для диспетчера.

Порядок работы (строго по шагам, каждый шаг — вызов инструмента):
1. fetch_weather — получить прогноз погоды Open-Meteo на 48 ч;
2. prepare_data — проверки входа (пропуски, диапазоны) и признаки модели;
3. run_model — прогноз мощности P10/P50/P90;
4. analyze_forecast — проверки и показатели: энергия, рампы, уверенность, климатология, ревизия к прошлому выпуску, флаги;
5. если есть флаги уровня warning — реши:
   * recalculate — перепроверить вход повторным запросом (вызови инструмент recalculate): при проблемах данных
     или существенной ревизии, когда нужно убедиться, что использован актуальный прогноз погоды;
   * flag — прогноз корректен, но диспетчеру нужно обратить внимание (рампы, ревизия, низкая уверенность);
   * accept — флагов уровня warning нет;
6. write_report — финальный шаг: decision (accept|recalculate|flag), reasoning (1–2 предложения, почему такое решение),
   narrative_ru (3–6 предложений на русском для диспетчера с конкретными числами из результатов инструментов:
   энергия за 48 ч и по суткам в часах номинала, средняя мощность, сравнение с климатической нормой, рампы, ревизия, уверенность).

Правила: используй только числа из результатов инструментов, ничего не выдумывай; не повторяй инструменты без причины;
всего не более 8 шагов. В режиме replay прогнозы погоды архивные — пересчёт покажет, что вход не изменился, это нормально.
Если инструмент вернул error — учти это в решении и в отчёте."""


def _fn(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {}, "required": required or [], "additionalProperties": False}}}


TOOL_SCHEMAS = [
    _fn("fetch_weather", "Получить прогноз погоды Open-Meteo на 48 ч для обеих турбин: в replay — архив запусков, "
                         "доступных на момент выпуска (Previous Runs), в live — последний оперативный запуск."),
    _fn("prepare_data", "Проверить вход (пропуски, скорости 0–60 м/с, температура −50…50 °C) и построить признаки модели."),
    _fn("run_model", "Запустить модель мощности: почасовые P10/P50/P90 (доля номинала) для t1, t2 и парка на 48 ч."),
    _fn("analyze_forecast", "Проверить прогноз и посчитать показатели: энергия (часы номинала) за 48 ч и по суткам, "
                            "доли часов ≥0.9 и ≤0.05, рампы, ширина P90−P10, климатология, ревизия к прошлому выпуску, флаги."),
    _fn("recalculate", "Повторно запросить погоду и пересчитать прогноз; сравнить хэш входа со старым.",
        {"reason": {"type": "string", "description": "почему нужен пересчёт"}}, ["reason"]),
    _fn("write_report", "Финальный шаг: записать отчёт и прогноз с решением агента.",
        {"decision": {"type": "string", "enum": list(DECISIONS)},
         "reasoning": {"type": "string", "description": "1–2 предложения: почему такое решение"},
         "narrative_ru": {"type": "string", "description": "3–6 предложений на русском для диспетчера с числами из анализа"}},
        ["decision", "reasoning", "narrative_ru"]),
]


def llm_settings() -> dict | None:
    """Настройки LLM из окружения или None, если ключа нет (значение ключа нигде не логируется)."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return {"model": os.getenv("OPENAI_MODEL") or DEFAULT_MODEL,
            "reasoning_effort": os.getenv("OPENAI_REASONING_EFFORT", "low")}


def _request_kwargs(model: str, reasoning_effort: str | None) -> dict:
    """Параметры запроса: temperature/max_tokens не передаём (gpt-5-* их не принимает, лимит съел бы reasoning),
    reasoning_effort — только для reasoning-моделей (gpt-5*, o*)."""
    kw = {}
    if reasoning_effort and (model.startswith("gpt-5") or re.match(r"^o\d", model)):
        kw["reasoning_effort"] = reasoning_effort
    return kw


def make_client():
    from openai import OpenAI  # импорт здесь: без ключа SDK не нужен
    return OpenAI(timeout=TIMEOUT_S, max_retries=1)


def validate_final(args: dict) -> tuple[dict | None, str]:
    """Проверить финальный ответ LLM: decision ∈ {accept, recalculate, flag}, reasoning, narrative_ru на русском с числами."""
    if not isinstance(args, dict):
        return None, "ожидался JSON-объект"
    errors = []
    decision = str(args.get("decision", "")).strip().lower()
    if decision not in DECISIONS:
        errors.append(f"decision должен быть одним из {DECISIONS}")
    reasoning = clean_text(args.get("reasoning", ""), 800)
    if len(reasoning) < 10:
        errors.append("reasoning пустой или слишком короткий")
    narrative = clean_text(args.get("narrative_ru", ""), 2000)
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", narrative)
    cyr = sum(1 for ch in letters if re.match(r"[А-Яа-яЁё]", ch))
    n_sent = len([s for s in re.split(r"(?<=[.!?])\s+", narrative) if len(s.strip()) > 3])
    if len(narrative) < 80:
        errors.append("narrative_ru слишком короткий (нужно 3–6 предложений)")
    elif not letters or cyr / len(letters) < 0.6:
        errors.append("narrative_ru должен быть на русском")
    elif not re.search(r"\d", narrative):
        errors.append("narrative_ru должен содержать числа из анализа")
    elif not 2 <= n_sent <= 10:          # с запасом: сокращения вида «ч.н.» режут предложения
        errors.append(f"narrative_ru: {n_sent} предложений, нужно 3–6")
    if errors:
        return None, "; ".join(errors)
    return {"decision": decision, "reasoning": reasoning, "narrative_ru": narrative}, ""


def parse_json_text(text: str) -> dict | None:
    """Вытащить JSON-объект из текстового ответа (на случай, если LLM ответила текстом, а не вызовом write_report)."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def run_llm_cycle(session, client, model: str, context: str, max_steps: int = MAX_STEPS,
                  reasoning_effort: str | None = "low") -> bool:
    """Агентный цикл: LLM вызывает инструменты сессии, пока не выполнит write_report (не более max_steps обращений).

    Возвращает True, если LLM довела цикл до валидного write_report; False — шаги кончились (тогда
    оркестратор завершит цикл по правилам). Сетевые/API-ошибки пробрасываются наверх.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": context}]
    extra = _request_kwargs(model, reasoning_effort)
    text_turns = 0
    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(model=model, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto", **extra)
        msg = resp.choices[0].message
        calls = list(msg.tool_calls or [])
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "total_tokens", None) if usage is not None else None
        session.log_llm_turn(step, time.perf_counter() - t0, [c.function.name for c in calls], tokens)
        if calls:
            text_turns = 0
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError
                except (json.JSONDecodeError, ValueError):
                    result = {"error": "аргументы инструмента должны быть JSON-объектом"}
                else:
                    result = session.dispatch(c.function.name, args)
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "content": json.dumps(result, ensure_ascii=False, default=str)})
            if session.final is not None:
                return True
            continue
        # ответ текстом: пустой — ошибка шага (fallback); иначе пробуем принять как финальный JSON
        if not (msg.content or "").strip():
            log.warning("LLM вернула пустой ответ без вызова инструментов (ход %d) — завершаю по правилам", step)
            return False
        text_turns += 1
        if text_turns >= 2:
            log.warning("LLM дважды подряд ответила текстом без валидного результата — завершаю по правилам")
            return False
        messages.append({"role": "assistant", "content": msg.content or ""})
        parsed = parse_json_text(msg.content or "")
        if parsed is not None:
            result = session.dispatch("write_report", parsed)
            if session.final is not None:
                return True
            messages.append({"role": "user", "content": f"Ответ не прошёл валидацию: {result.get('error')}. "
                                                        "Вызови write_report с корректными полями."})
        else:
            messages.append({"role": "user", "content": "Продолжай по порядку шагов и заверши цикл вызовом write_report "
                                                        "(decision, reasoning, narrative_ru)."})
    log.warning("LLM не завершила цикл за %d шагов — завершаю по правилам", max_steps)
    return False
