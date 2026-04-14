# Отчет по сравнению

## Рейтинг
- nbeats: WAPE=0.0445, разница_с_лидером=0.0000, разница_с_seasonal_naive=-0.1338
- mlp: WAPE=0.0594, разница_с_лидером=0.0149, разница_с_seasonal_naive=-0.1189
- ets: WAPE=0.0822, разница_с_лидером=0.0378, разница_с_seasonal_naive=-0.0960
- lightgbm: WAPE=0.0863, разница_с_лидером=0.0419, разница_с_seasonal_naive=-0.0919
- catboost: WAPE=0.0930, разница_с_лидером=0.0485, разница_с_seasonal_naive=-0.0853
- ridge: WAPE=0.0945, разница_с_лидером=0.0500, разница_с_seasonal_naive=-0.0838
- naive: WAPE=0.1081, разница_с_лидером=0.0636, разница_с_seasonal_naive=-0.0702
- seasonal_naive: WAPE=0.1783, разница_с_лидером=0.1338, разница_с_seasonal_naive=0.0000

## Устойчивость по сегментам
- stable: best=nbeats, WAPE=0.0445

## Бутстрэп
- catboost - ets: средняя_разность=0.0149, 95% ДИ=[-0.0223, 0.0475]
- catboost - lightgbm: средняя_разность=0.0071, 95% ДИ=[0.0031, 0.0106]
- catboost - mlp: средняя_разность=0.0349, 95% ДИ=[0.0235, 0.0449]
- catboost - naive: средняя_разность=-0.0116, 95% ДИ=[-0.0428, 0.0158]
- catboost - nbeats: средняя_разность=0.0480, 95% ДИ=[0.0442, 0.0524]
- catboost - ridge: средняя_разность=-0.0032, 95% ДИ=[-0.0160, 0.0114]
- catboost - seasonal_naive: средняя_разность=-0.0822, 95% ДИ=[-0.1094, -0.0584]
- ets - lightgbm: средняя_разность=-0.0078, 95% ДИ=[-0.0369, 0.0254]
- ets - mlp: средняя_разность=0.0200, 95% ДИ=[-0.0027, 0.0458]
- ets - naive: средняя_разность=-0.0265, 95% ДИ=[-0.0318, -0.0206]
- ets - nbeats: средняя_разность=0.0331, 95% ДИ=[-0.0034, 0.0747]
- ets - ridge: средняя_разность=-0.0181, 95% ДИ=[-0.0635, 0.0337]
- ets - seasonal_naive: средняя_разность=-0.0971, 95% ДИ=[-0.1060, -0.0871]
- lightgbm - mlp: средняя_разность=0.0278, 95% ДИ=[0.0205, 0.0343]
- lightgbm - naive: средняя_разность=-0.0187, 95% ДИ=[-0.0459, 0.0052]
- lightgbm - nbeats: средняя_разность=0.0409, 95% ДИ=[0.0336, 0.0493]
- lightgbm - ridge: средняя_разность=-0.0102, 95% ДИ=[-0.0266, 0.0084]
- lightgbm - seasonal_naive: средняя_разность=-0.0893, 95% ДИ=[-0.1125, -0.0690]
- mlp - naive: средняя_разность=-0.0465, 95% ДИ=[-0.0664, -0.0291]
- mlp - nbeats: средняя_разность=0.0131, 95% ДИ=[-0.0007, 0.0289]
- mlp - ridge: средняя_разность=-0.0381, 95% ДИ=[-0.0608, -0.0121]
- mlp - seasonal_naive: средняя_разность=-0.1171, 95% ДИ=[-0.1329, -0.1033]
- naive - nbeats: средняя_разность=0.0596, 95% ДИ=[0.0284, 0.0952]
- naive - ridge: средняя_разность=0.0085, 95% ДИ=[-0.0317, 0.0543]
- naive - seasonal_naive: средняя_разность=-0.0706, 95% ДИ=[-0.0742, -0.0665]
- nbeats - ridge: средняя_разность=-0.0512, 95% ДИ=[-0.0601, -0.0410]
- nbeats - seasonal_naive: средняя_разность=-0.1302, 95% ДИ=[-0.1618, -0.1026]
- ridge - seasonal_naive: средняя_разность=-0.0791, 95% ДИ=[-0.1208, -0.0425]

## Rolling Vs Holdout
- При допущении конечной дисперсии потерь по forecast origin средняя rolling-оценка по K окнам имеет дисперсию порядка sigma^2 / K, тогда как одиночный holdout соответствует K = 1.
- nbeats: rolling_WAPE=0.0445, holdout_last_origin=0.0445, std_WAPE_по_origin=nan, SE_rolling=nan, фактор_снижения_дисперсии~1.0
- mlp: rolling_WAPE=0.0594, holdout_last_origin=0.0594, std_WAPE_по_origin=nan, SE_rolling=nan, фактор_снижения_дисперсии~1.0
- ets: rolling_WAPE=0.0822, holdout_last_origin=0.0822, std_WAPE_по_origin=nan, SE_rolling=nan, фактор_снижения_дисперсии~1.0
- lightgbm: rolling_WAPE=0.0863, holdout_last_origin=0.0863, std_WAPE_по_origin=nan, SE_rolling=nan, фактор_снижения_дисперсии~1.0
- catboost: rolling_WAPE=0.0930, holdout_last_origin=0.0930, std_WAPE_по_origin=nan, SE_rolling=nan, фактор_снижения_дисперсии~1.0
- ridge: rolling_WAPE=0.0945, holdout_last_origin=0.0945, std_WAPE_по_origin=nan, SE_rolling=nan, фактор_снижения_дисперсии~1.0
- naive: rolling_WAPE=0.1081, holdout_last_origin=0.1081, std_WAPE_по_origin=nan, SE_rolling=nan, фактор_снижения_дисперсии~1.0
- seasonal_naive: rolling_WAPE=0.1783, holdout_last_origin=0.1783, std_WAPE_по_origin=nan, SE_rolling=nan, фактор_снижения_дисперсии~1.0

## Финальная рекомендация
- выбранная модель: nbeats
- метрика выбора: WAPE=0.0445
- улучшение_относительно_seasonal_naive: 0.1338
- область применимости: рекомендация валидна только для проверенных rolling-окон, принятой feature policy и текущей настройки датасета
