# Сводка запуска

## Что запускалось
- уровень агрегации: category
- набор признаков: F4
- число окон backtesting: 2
- seed: 42
- полное время пайплайна, сек.: 16.192

## Что было проверено
- experiment_dataset_bundling: выполнено | wall_time=2.944 сек.
- experiment_dataset_preparation: выполнено | wall_time=6.657 сек.
- experiment_segmentation: выполнено | wall_time=0.002 сек.
- experiment_feature_registry: выполнено | wall_time=0.011 сек.
- experiment_supervised_building: выполнено | wall_time=0.022 сек.
- experiment_backtest_window_generation: выполнено | wall_time=0.003 сек.
- experiment_model_execution: выполнено | wall_time=6.239 сек.
- experiment_comparison: выполнено | wall_time=0.033 сек.
- experiment_best_model_fit: выполнено | wall_time=0.280 сек.

## Модели
- naive: family=baseline | feature_count=1 | features=enabled
- seasonal_naive: family=baseline | feature_count=2 | features=enabled
- ets: family=statistical | feature_count=0 | features=disabled
- ridge: family=ml | feature_count=19 | features=enabled
- lightgbm: family=ml | feature_count=19 | features=enabled
- catboost: family=ml | feature_count=19 | features=enabled
- mlp: family=dl | feature_count=0 | features=disabled
- nbeats: family=dl | feature_count=0 | features=disabled

## Что получилось
- итоговый артефакт финальной модели: models/best_model
- место 1: nbeats | WAPE=0.0445 | sMAPE=4.4347 | MAE=9526.5312 | Bias=-3763.3281
- место 2: mlp | WAPE=0.0594 | sMAPE=5.9936 | MAE=12712.9688 | Bias=-12712.9688
- место 3: ets | WAPE=0.0822 | sMAPE=8.4239 | MAE=17615.7129 | Bias=-17615.7129
- место 4: lightgbm | WAPE=0.0863 | sMAPE=8.9174 | MAE=18492.4407 | Bias=-18492.4407
- место 5: catboost | WAPE=0.0930 | sMAPE=9.6568 | MAE=19912.6717 | Bias=-19912.6717
- место 6: ridge | WAPE=0.0945 | sMAPE=9.8865 | MAE=20238.6389 | Bias=-20238.6389
- место 7: naive | WAPE=0.1081 | sMAPE=11.2732 | MAE=23154.5000 | Bias=-23154.5000
- место 8: seasonal_naive | WAPE=0.1783 | sMAPE=19.4101 | MAE=38179.5000 | Bias=-38179.5000
