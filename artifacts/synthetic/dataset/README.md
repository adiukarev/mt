# Synthetic Weekly Dataset

## Что внутри
- Рядов: 96
- Сценарии: stable, seasonal_promo, noisy_stockout
- Категорий: foods, household, hobbies
- История: 208 недель
- Горизонт future actual: 8 недель
- Всего строк: 62208
- Средняя продажа по ряду: 158.96

## Классы ковариат
- `promo_planned`: known-in-advance
- `price`: known-in-advance
- `sales_units`: target / observed over time
- `expected_sales_units`: скрытая вспомогательная величина генератора, не для честного forecast-контура

## Ограничения
- Спрос синтетический и задан формулой генератора, поэтому он проще реального retail-процесса.
- Промо и цена считаются заранее известными по определению манифеста.
- Не моделируются реальные межтоварные замещения, supply-chain лаги, календарь праздников и иерархия M5.
- Preview-каталог содержит audit-like артефакты и графики synthetic данных.

## Распределение рядов по категориям
- noisy_stockout / foods: 32
- noisy_stockout / hobbies: 32
- noisy_stockout / household: 32
- seasonal_promo / foods: 32
- seasonal_promo / hobbies: 32
- seasonal_promo / household: 32
- stable / foods: 32
- stable / hobbies: 32
- stable / household: 32
