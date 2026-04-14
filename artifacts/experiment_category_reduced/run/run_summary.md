# Сводка запуска

## Что запускалось
- уровень агрегации: category
- набор признаков: F4
- число окон backtesting: 256
- seed: 42
- полное время пайплайна, сек.: 749.555

## Что было проверено
- experiment_dataset_bundling: выполнено | wall_time=3.345 сек.
- experiment_dataset_preparation: выполнено | wall_time=6.709 сек.
- experiment_segmentation: выполнено | wall_time=0.015 сек.
- experiment_feature_registry: выполнено | wall_time=0.005 сек.
- experiment_supervised_building: выполнено | wall_time=0.033 сек.
- experiment_backtest_window_generation: выполнено | wall_time=0.013 сек.
- experiment_model_execution: выполнено | wall_time=732.245 сек.
- experiment_comparison: выполнено | wall_time=0.096 сек.
- experiment_best_model_fit: выполнено | wall_time=7.094 сек.

## Модели
- naive: family=baseline | feature_count=1 | features=enabled
- seasonal_naive: family=baseline | feature_count=2 | features=enabled
- ets: family=statistical | feature_count=0 | features=disabled
- ridge: family=ml | feature_count=44 | features=enabled
- lightgbm: family=ml | feature_count=44 | features=enabled
- catboost: family=ml | feature_count=44 | features=enabled
- mlp: family=dl | feature_count=0 | features=disabled
- nbeats: family=dl | feature_count=0 | features=disabled

## Что получилось
- итоговый артефакт финальной модели: models/best_model
- место 1: nbeats | WAPE=0.0506 | sMAPE=5.2868 | MAE=4394.5694 | Bias=328.0902
- место 2: lightgbm | WAPE=0.0516 | sMAPE=5.4304 | MAE=4485.3324 | Bias=-1106.1292
- место 3: mlp | WAPE=0.0521 | sMAPE=5.2406 | MAE=4524.1620 | Bias=434.1941
- место 4: catboost | WAPE=0.0601 | sMAPE=6.6435 | MAE=5223.1837 | Bias=-78.8357
- место 5: ridge | WAPE=0.0616 | sMAPE=7.9548 | MAE=5357.6133 | Bias=-1187.3528
- место 6: naive | WAPE=0.0757 | sMAPE=6.7489 | MAE=6582.4076 | Bias=-101.1263
- место 7: ets | WAPE=0.0777 | sMAPE=7.3120 | MAE=6754.9104 | Bias=543.6817
- место 8: seasonal_naive | WAPE=0.1007 | sMAPE=11.8875 | MAE=8755.3190 | Bias=-3293.3867
