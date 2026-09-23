"""Small event-driven feedback for the operator panel; no replay of completed work."""
from __future__ import annotations

from html import escape

_STAGES = (
    ("Погода", ("fetch_weather",)),
    ("Данные", ("prepare_data", "recalculate")),
    ("Прогноз", ("run_model",)),
    ("Проверка", ("analyze_forecast", "llm", "verify_narrative", "write_report")),
)
_ALIASES = {"done": "ok", "success": "ok", "completed": "ok", "skipped": "skip", "run": "running",
            "failed": "error", "warn": "warning"}
_STATES = {"ok": ("✓", "готово"), "pending": ("·", "ожидает"), "skip": ("−", "пропущено"),
           "running": ("…", "в работе"), "warning": ("!", "есть ограничения"), "error": ("×", "ошибка")}
_CSS = """
<style>
.wa-progress{padding:12px 14px;border:1px solid color-mix(in srgb,currentColor 20%,transparent);border-radius:12px}
.wa-progress-title{font-size:13px;font-weight:600;margin:0 0 10px}
.wa-progress[data-running="true"] .wa-progress-title::before{content:"";display:inline-block;width:6px;height:6px;margin-right:7px;vertical-align:1px;border-radius:50%;background:currentColor;animation:wa-working 1s ease-in-out infinite alternate}
.wa-progress ol{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:0;padding:0;list-style:none}
.wa-progress li{display:flex;gap:6px;align-items:center;min-width:0;font-size:12px;line-height:20px}
.wa-progress .wa-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.wa-progress .wa-mark{display:inline-flex;justify-content:center;align-items:center;width:18px;height:18px;flex:0 0 18px;border-radius:50%;border:1px solid currentColor;font-size:12px;line-height:1}
.wa-progress [data-state="pending"] .wa-mark,.wa-progress [data-state="skip"] .wa-mark{border-style:dashed;opacity:.7}
.wa-progress [data-state="ok"] .wa-mark{color:light-dark(#087f6b,#6ad9c5)}
.wa-progress [data-state="warning"] .wa-mark{color:light-dark(#9a5a00,#efc36b)}
.wa-progress [data-state="error"] .wa-mark{color:light-dark(#b42338,#ff91a2)}
.wa-progress [data-state="running"] .wa-mark{animation:wa-working 1s ease-in-out infinite alternate}
.wa-sr{position:absolute;width:1px;height:1px;padding:0;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@keyframes wa-working{from{opacity:.45}to{opacity:1}}
@media(max-width:420px){.wa-progress{padding:10px 12px}.wa-progress ol{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px 20px}.wa-progress li{font-size:12px;gap:7px}.wa-progress .wa-mark{width:18px;height:18px;flex-basis:18px}}
@media(prefers-reduced-motion:reduce){.wa-progress [data-state="running"] .wa-mark,.wa-progress[data-running="true"] .wa-progress-title::before{animation:none}}
</style>
"""


def _stage_state(tools: tuple[str, ...], latest: dict[str, str]) -> str:
    values = [latest[t] for t in tools if t in latest]
    if not values:
        return "pending"
    for state in ("error", "running", "warning", "pending", "skip"):
        if state in values:
            return state
    return "ok" if all(s == "ok" for s in values) else "pending"


def progress_markup(steps: list[dict] | None = None, *, running: bool = True,
                    active_tool: str | None = None) -> str:
    """Render only observed states. Set active_tool when a tool actually starts.

    Each step contains tool/status, as in ForecastResult.trace. Calls with
    running=False never pulse, even if a stale trace contains a running step.
    Unknown or absent status means pending. A later record supersedes an earlier
    record for the same tool (e.g. an explicit successful retry).
    """
    latest = {str(s.get("tool", "")): _ALIASES.get(str(s.get("status", "pending")),
              str(s.get("status", "pending"))) for s in steps or []}
    if running and active_tool:
        latest[active_tool] = "running"
    if not running:
        latest = {tool: "pending" if state == "running" else state for tool, state in latest.items()}
    states = [_stage_state(tools, latest) for _, tools in _STAGES]
    title = "Идёт расчёт" if running else ("Нужна проверка" if "error" in states else
            "Есть ограничения" if any(s in {"warning", "skip"} for s in states) else
            "Расчёт готов" if all(s == "ok" for s in states) else "Ожидает расчёта")
    items = []
    for (label, _), state in zip(_STAGES, states):
        mark, description = _STATES[state]
        items.append(f'<li data-state="{state}" title="{escape(label)}: {description}">'
                     f'<span class="wa-mark" aria-hidden="true">{mark}</span>'
                     f'<span class="wa-label">{escape(label)}</span>'
                     f'<span class="wa-sr">: {description}</span></li>')
    return (_CSS + f'<section class="wa-progress" data-running="{str(running).lower()}" aria-label="Ход расчёта WindAgent">'
            f'<p class="wa-progress-title" role="status">{title}</p><ol>{"".join(items)}</ol></section>')


def mode_activation_markup() -> str:
    """One short acknowledgement; insert only for an explicit off→on action."""
    return """<style>
.wa-mode-ack{position:relative;isolation:isolate;display:inline-block;overflow:hidden;border-radius:7px;padding:5px 9px;font-size:12px;line-height:20px;color:inherit}
.wa-mode-ack::before{content:"";position:absolute;z-index:-1;inset:0;border-radius:inherit;background:linear-gradient(100deg,transparent,rgba(151,112,239,.25),transparent);animation:wa-mode-on .9s ease-out 1 both}
@keyframes wa-mode-on{0%{transform:translateX(-110%);opacity:0}20%{opacity:1}100%{transform:translateX(110%);opacity:0}}
@media(prefers-reduced-motion:reduce){.wa-mode-ack::before{animation:none;display:none}}
</style><span class="wa-mode-ack" role="status">AI-пояснения включены</span>"""
