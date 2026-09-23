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

    # площадка: по умолчанию ВЭС «Нурлы» (T1, T2, модель обучена на её истории); любая другая — по координатам,
    # прогноз переносом модели (без истории площадки), результаты в outputs/adhoc/<площадка>/
    site_p = argparse.ArgumentParser(add_help=False)
    site_p.add_argument("--lat", type=float, help="широта площадки (вместе с --lon: прогноз для своей площадки)")
    site_p.add_argument("--lon", type=float, help="долгота площадки")
    site_p.add_argument("--n-turbines", type=int, default=1, help="число турбин площадки (для МВт·ч)")
    site_p.add_argument("--rated-mw", type=float, default=None, help="номинал одной турбины, МВт (для МВт·ч)")
    site_p.add_argument("--site-name", default=None, help="название площадки в отчёте")
    site_p.add_argument("--issue-hour", type=int, default=None, choices=range(24), metavar="0-23",
                        help=f"час выпуска местного времени (по умолчанию {config.ISSUE_HOUR_LOCAL} — протокол ТЗ)")

    r = sub.add_parser("replay", parents=[common, site_p], help="воспроизвести ежедневные прогнозы тестового периода как в прошлом")
    r.add_argument("--start", default=config.TEST_ISSUE_START)
    r.add_argument("--end", default=config.TEST_ISSUE_END)
    r.add_argument("--no-llm", action="store_true", help="детерминированный режим без LLM")

    f = sub.add_parser("forecast", parents=[common, site_p], help="один прогноз на дату выпуска (архивный прогноз погоды)")
    f.add_argument("--issue-date", required=True)
    f.add_argument("--no-llm", action="store_true")

    lv = sub.add_parser("live", parents=[common, site_p], help="оперативный прогноз по текущему прогнозу погоды")
    lv.add_argument("--no-llm", action="store_true")

    ev = sub.add_parser("evaluate", parents=[common], help="сверить прогнозы с фактом (полный файл организаторов) → MAE/RMSE по горизонтам")
    ev.add_argument("--actual", required=True, nargs="+", help="CSV с фактом в формате датасета (один на турбину: t1, t2)")
    ev.add_argument("--turbines", nargs="+", default=["t1", "t2"], help="ключи турбин в порядке файлов --actual")
    ev.add_argument("--forecasts", default=None, help="папка с прогнозами (по умолчанию outputs/forecasts)")

    sb = sub.add_parser("submission", parents=[common], help="собрать единый файл сдачи outputs/submission_feb2026.csv из outputs/forecasts")
    sb.add_argument("--forecasts", default=None)

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
    elif a.cmd == "evaluate":
        from .evaluate import evaluate_cli
        return evaluate_cli(a.actual, a.turbines, a.forecasts)
    elif a.cmd == "submission":
        from .evaluate import build_submission
        build_submission(a.forecasts)
    else:
        from .agent.orchestrator import run_live, run_replay
        use_llm = not a.no_llm
        site = None
        if a.lat is not None or a.lon is not None:
            if a.lat is None or a.lon is None:
                p.error("нужны обе координаты: --lat и --lon")
            site = config.custom_site(a.lat, a.lon, n_turbines=a.n_turbines, rated_mw=a.rated_mw, name=a.site_name)
        if a.cmd in ("replay", "forecast"):
            start, end = (a.start, a.end) if a.cmd == "replay" else (a.issue_date, a.issue_date)
            summary = run_replay(wx, start, end, use_llm=use_llm, site=site, issue_hour=a.issue_hour)
            n_err = int((summary["status"] == "error").sum()) if len(summary) else 0
            if n_err:   # код выхода ≠ 0, чтобы автоматическая проверка заметила упавшие выпуски
                logging.getLogger(__name__).error("выпусков с ошибкой: %d из %d", n_err, len(summary))
                return 1
        elif a.cmd == "live":
            if run_live(wx, use_llm=use_llm, site=site) is None:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
