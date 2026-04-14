# Сводка запуска

## Что запускалось
- уровень агрегации: category
- набор признаков: F4
- число окон backtesting: 1620
- seed: 42
- полное время пайплайна, сек.: 5981.979

## Что было проверено
- experiment_dataset_bundling: выполнено | wall_time=2.935 сек.
- experiment_dataset_preparation: выполнено | wall_time=6.577 сек.
- experiment_segmentation: выполнено | wall_time=0.003 сек.
- experiment_feature_registry: выполнено | wall_time=0.006 сек.
- experiment_supervised_building: выполнено | wall_time=0.033 сек.
- experiment_backtest_window_generation: выполнено | wall_time=0.067 сек.
- experiment_model_execution: выполнено | wall_time=5968.156 сек.
- experiment_comparison: выполнено | wall_time=0.405 сек.
- experiment_best_model_fit: выполнено | wall_time=3.796 сек.

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
- место 1: mlp | WAPE=0.0650 | sMAPE=7.1807 | MAE=5545.6889 | Bias=671.1944
- место 2: nbeats | WAPE=0.0652 | sMAPE=7.3166 | MAE=5570.0301 | Bias=578.0665
- место 3: catboost | WAPE=0.0717 | sMAPE=8.5548 | MAE=6123.6498 | Bias=381.7068
- место 4: lightgbm | WAPE=0.0720 | sMAPE=8.3585 | MAE=6143.6153 | Bias=-665.6415
- место 5: ridge | WAPE=0.0737 | sMAPE=10.9314 | MAE=6290.8637 | Bias=-874.3907
- место 6: ets | WAPE=0.0838 | sMAPE=8.3919 | MAE=7156.4957 | Bias=656.4083
- место 7: naive | WAPE=0.0851 | sMAPE=8.1328 | MAE=7267.0002 | Bias=-528.6677
- место 8: seasonal_naive | WAPE=0.1343 | sMAPE=16.0819 | MAE=11467.4735 | Bias=-6817.7595
