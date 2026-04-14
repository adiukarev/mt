# Synthetic Prediction Report

- Модель: nbeats
- Сценарии: noisy_stockout, seasonal_promo, stable
- Рядов с прогнозом: 3
- Forecast rows: 72

## Ограничения
- В качестве первичного источника используется saved `model.pkl`; при невозможности прямого synthetic inference pipeline откатывается к artifact-guided refit того же типа модели.
- Текущая команда ожидает колонку `is_history`, чтобы честно разделить history и future actual.
- Overlay отражает один выбранный ряд; для остальных рядов сохраняется только CSV-прогноз и метрики.
- Источник модели: saved artifact `artifacts/experiment_category_reduced/models/best_model`

## Scenario coverage
- noisy_stockout: 3 series, 648 rows
- seasonal_promo: 3 series, 648 rows
- stable: 3 series, 648 rows

## Best diagnostic row
- scenario_name: stable
- horizon: 2
- WAPE: 0.0211
