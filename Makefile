# Короткие команды для жюри и команды. Запускать из корня репозитория.
# Без ключа OpenAI всё работает детерминированно; сеть нужна только для установки зависимостей
# (кэш Open-Meteo и обученная модель лежат в репозитории).

PYTHON ?= $(shell for p in python3.11 python3.12 python3.13 python3; do command -v $$p >/dev/null 2>&1 && { echo $$p; break; }; done)
VENV   ?= .venv
BIN    := $(VENV)/bin
WA     := $(BIN)/wind-agent
START  ?= 2026-01-31
END    ?= 2026-02-27
DATE   ?= 2026-02-10

.PHONY: help setup backtest train replay replay-fast forecast live test viz docker docker-replay clean clean-outputs

help:
	@echo "make setup        создать $(VENV) и установить проект (uv, если есть; иначе python -m venv + pip)"
	@echo "make test         pytest, офлайн, < 1 мин"
	@echo "make replay-fast  3 даты выпуска (31.01-02.02.2026), без LLM, без сети"
	@echo "make replay       все даты выпуска $(START)..$(END) (LLM, если в .env есть OPENAI_API_KEY)"
	@echo "make forecast     один выпуск: make forecast DATE=2026-02-10"
	@echo "make backtest     честный бэктест дек.2025-янв.2026 -> outputs/backtest_metrics.json"
	@echo "make train        переобучить финальную модель -> models/power_model.joblib"
	@echo "make live         оперативный прогноз по текущему прогнозу погоды (нужна сеть)"
	@echo "make viz          данные для 3D-визуализации рельефа и поля ветра"
	@echo "make docker       собрать Docker-образ wind-agent; make docker-replay - прогон в контейнере"
	@echo "make clean        удалить кэши Python/pytest (результаты и модель не трогает)"
	@echo "make clean-outputs удалить результаты прогонов агента (восстанавливаются make replay)"

setup:
	@if command -v uv >/dev/null 2>&1; then \
	    echo ">> uv: $(VENV) (Python 3.11)"; \
	    [ -d $(VENV) ] || uv venv --python 3.11 $(VENV); \
	    uv pip install --python $(BIN)/python -e ".[dev]" -c requirements.lock; \
	else \
	    echo ">> python -m venv + pip: $(VENV)"; \
	    [ -d $(VENV) ] || $(PYTHON) -m venv $(VENV); \
	    $(BIN)/python -m pip install --upgrade pip; \
	    $(BIN)/python -m pip install -e ".[dev]" -c requirements.lock; \
	fi
	@$(WA) --help >/dev/null && echo ">> готово: $(WA)"

backtest:
	$(WA) backtest

train:
	$(WA) train

replay:
	$(WA) replay --start $(START) --end $(END)

replay-fast:
	$(WA) --offline replay --no-llm --start 2026-01-31 --end 2026-02-02

forecast:
	$(WA) forecast --issue-date $(DATE)

live:
	$(WA) live

test:
	$(BIN)/python -m pytest -q

viz:
	$(BIN)/python scripts/fetch_dem.py
	$(BIN)/python scripts/build_viz_data.py

docker:
	docker build -t wind-agent .

docker-replay:
	docker run --rm -v "$(CURDIR)/outputs:/app/outputs" wind-agent --offline replay --no-llm

clean:
	rm -rf .pytest_cache outputs/tmp
	find . -path ./$(VENV) -prune -o -name __pycache__ -type d -exec rm -rf {} +

clean-outputs:
	rm -rf outputs/forecasts outputs/reports outputs/live outputs/agent_log.jsonl outputs/replay_summary.csv
