"""Bind a completed historical backtest to the matching refitted model and sources.

Run after `wind-agent backtest --offline` and `wind-agent train --offline` in
one unchanged checkout. The manifest describes a configuration validation;
it does not claim that the refitted artifact was held out from test labels.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from wind_agent import config
from wind_agent.model import PowerModel, FEATURE_CONTEXT_VERSION


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    model_path = config.MODELS_DIR / 'power_model.joblib'
    metrics_path = config.OUTPUTS_DIR / 'backtest_metrics.json'
    model = PowerModel.load(model_path)
    result = json.loads(metrics_path.read_text())
    if model.meta.get('feature_context') != FEATURE_CONTEXT_VERSION:
        raise SystemExit('Retrain the model with the current weather-context semantics first.')
    for key in ['feature_context', 'features', 'params']:
        if result.get(key) != model.meta.get(key):
            raise SystemExit(f'Backtest/model configuration mismatch: {key}; rerun backtest and train together.')
    if not model.calibration:
        raise SystemExit('Refusing to describe an uncalibrated model as the verified artifact.')
    sources = ['src/wind_agent/' + name + '.py' for name in ['data', 'features', 'model', 'config', 'weather', 'terrain']]
    manifest = {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'model_sha256': sha256(model_path),
        'metrics_file': str(metrics_path.relative_to(config.ROOT)),
        'metrics_sha256': sha256(metrics_path),
        'source_sha256': {p: sha256(config.ROOT / p) for p in sources},
        'protocol': 'Исторический временной бэктест конфигурации; финальная модель затем переобучена на всей истории.',
        'train_end': config.BACKTEST_TRAIN_END,
        'calibration_period': {'start': config.BACKTEST_CALIB_START, 'end': config.BACKTEST_TRAIN_END},
        'test_period': {'start': result['test_range'][0], 'end': result['test_range'][1]},
        'artifact_train_end': config.HISTORY_END,
        'artifact_calibration_period': {'start': config.BACKTEST_TEST_START, 'end': config.HISTORY_END},
        'feature_context': model.meta['feature_context'],
        'limitations': [
            'Это проверка конфигурации на декабре 2025–январе 2026, а не точность выбранного выпуска или текущего live-прогноза.',
            'Финальная модель переобучена, включая тестовые месяцы; метрики получены отдельной моделью с обучением до 30.11.2025.',
            'Две турбины и два зимних месяца; наблюдения по часам, турбинам и лагам зависимы; ограничения/простои отфильтрованы.',
            'Исправление погодного контекста повысило согласованность обучения и прогноза, но общая MAE на этом периоде выросла.',
            'Эмпирическое покрытие P10–P90 около 75%, ниже номинальных 80%; надёжность на новых сезонах и площадках не доказана.',
        ],
    }
    target = config.MODELS_DIR / 'validation_manifest.json'
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
