import numpy as np
import pandas as pd


def segment_series(weekly: pd.DataFrame) -> pd.DataFrame:
	"""Группировать ряды по поведению, чтобы потом анализировать серии"""

	rows: list[dict[str, object]] = []
	for series_id, group in weekly.groupby("series_id"):
		# Берем целевую переменную ряда и приводим к float для устойчивых численных расчетов
		sales = group["sales_units"].astype(float)

		# длина истории ряда в неделях
		history_len = int(len(group))
		# доля недель, где продажи были равны нулю
		zero_share = float((sales == 0).mean())
		# средний уровень продаж: защита на случай пустой группы
		mean_value = float(sales.mean()) if history_len else 0.0
		# cтандартное отклонение продаж по всей истории ряда.
		std_value = float(sales.std(ddof=0)) if history_len else 0.0
		# Коэффициент вариации - относительная шумность ряда, а не абсолютный разброс.
		cov = std_value / mean_value if mean_value else float("inf")
		# флаг короткой истории относительно годового цикла
		short_history = history_len < 52
		# флаг высокой доли нулей
		high_zero_share = zero_share >= 0.4
		# абсолютная дисперсия продаж
		variance = float(np.var(sales))
		# флаг высокой относительной вариативности
		# если среднее равно нулю, cov бесконечен и флаг тоже считаем true
		high_variance = cov > 1.0 if np.isfinite(cov) else True

		if history_len < 26:  # Очень короткие ряды помечаем как проблемные: статистика по ним ненадежна
			label = "problematic"
		elif zero_share >= 0.4:  # Если нулевых недель много, ряд считаем прерывистым
			label = "intermittent"
		elif cov <= 0.3:  # Низкий коэффициент вариации интерпретируем как стабильный ряд
			label = "stable"
		else:  # Все остальные ряды считаем умеренно шумными.
			label = "medium_noise"

		rows.append(
			{
				"series_id": series_id,
				"segment_label": label,
				"history_weeks": history_len,
				"zero_share": zero_share,
				"short_history": short_history,
				"high_zero_share": high_zero_share,
				"variance": variance,
				"high_variance": high_variance
			}
		)

	# одна строка = один ряд
	return pd.DataFrame(rows)
