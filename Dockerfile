# syntax=docker/dockerfile:1
# Образ с CLI wind-agent. Данные организаторов, кэш Open-Meteo и обученная модель уже лежат в репозитории,
# поэтому контейнер работает без сети и без ключей:
#   docker build -t wind-agent .
#   docker run --rm -v "$PWD/outputs:/app/outputs" wind-agent --offline replay --no-llm
# LLM-режим: docker run --rm --env-file .env -v "$PWD/outputs:/app/outputs" wind-agent replay
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Сначала метаданные пакета и исходники — слой с зависимостями кэшируется между сборками.
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
# Установка в режиме -e (editable): config.ROOT вычисляется от расположения src/wind_agent,
# поэтому data/, models/, outputs/ ищутся в /app. При обычном `pip install .` пакет попал бы
# в site-packages, и пути к данным указывали бы не туда.
RUN pip install -e . -c requirements.lock

# Остальной проект: data/raw, data/cache/openmeteo, models/, outputs/, scripts/ и т.д. (см. .dockerignore)
COPY . .

ENTRYPOINT ["wind-agent"]
CMD ["--help"]
