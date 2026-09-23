"""Командная строка: wind-agent train | backtest | replay | forecast | live."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

from . import config


def main(argv: list[str] | None = None) -> int:
    load_dotenv(config.ROOT / ".env")
    p = argparse.ArgumentParser(prog="wind-agent", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--offline", action="store_true", help="не ходить в Open-Meteo, использовать только кэш")
    sub = p.add_subparsers(dest="cmd", required=True)
    # те же флаги принимаются и после подкоманды: `wind-agent replay --offline --no-llm`
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--offline", action="store_true", default=argparse.SUPPRESS, help="только кэш Open-Meteo")

    sub.add_parser("backtest", parents=[common], help="честный бэктест на дек.2025–янв.2026 (есть факт)")
    sub.add_parser("train", parents=[common], help="обучить финальную модель на всей истории до 31.01.2026")

    r = sub.add_parser("replay", parents=[common], help="воспроизвести ежедневные прогнозы тестового периода как в прошлом")
    r.add_argument("--start", default=config.TEST_ISSUE_START)
    r.add_argument("--end", default=config.TEST_ISSUE_END)
    r.add_argument("--no-llm", action="store_true", help="детерминированный режим без LLM")

    f = sub.add_parser("forecast", parents=[common], help="один прогноз на дату выпуска (архивный прогноз погоды)")
    f.add_argument("--issue-date", required=True)
    f.add_argument("--no-llm", action="store_true")

    lv = sub.add_parser("live", parents=[common], help="оперативный прогноз по текущему прогнозу погоды")
    lv.add_argument("--no-llm", action="store_true")

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):   # HTTP-клиенты логируют каждый запрос — оставляем только предупреждения
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from .weather import WeatherClient
    wx = WeatherClient(offline=a.offline)

    if a.cmd == "backtest":
        from .model import backtest
        res = backtest(wx)
        out = {k: v["mae"] for k, v in res["overall"].items() if isinstance(v, dict)}
        out["coverage_p10_p90_pct"] = res["overall"].get("coverage_p10_p90_pct")
        print(json.dumps(out, indent=2))
    elif a.cmd == "train":
        from .model import train_final
        m = train_final(wx)
        print(json.dumps(m.meta, ensure_ascii=False, indent=2, default=str))
    else:
        from .agent.orchestrator import run_live, run_replay
        use_llm = not a.no_llm
        if a.cmd in ("replay", "forecast"):
            start, end = (a.start, a.end) if a.cmd == "replay" else (a.issue_date, a.issue_date)
            summary = run_replay(wx, start, end, use_llm=use_llm)
            n_err = int((summary["status"] == "error").sum()) if len(summary) else 0
            if n_err:   # код выхода ≠ 0, чтобы автоматическая проверка заметила упавшие выпуски
                logging.getLogger(__name__).error("выпусков с ошибкой: %d из %d", n_err, len(summary))
                return 1
        elif a.cmd == "live":
            if run_live(wx, use_llm=use_llm) is None:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
