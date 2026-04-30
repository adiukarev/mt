from matplotlib.axes import Axes

PLOT_LABELS: dict[str, str] = {
	"date": "Дата",
	"week_start": "Начало недели",
	"month_start": "Начало месяца",
	"horizon": "Горизонт прогноза, нед.",
	"model_name": "Модель",
	"segment": "Сегмент",
	"segment_label": "Сегмент",
	"category": "Категория",
	"series_id": "Идентификатор ряда",
	"scenario_name": "Сценарий",
	"sales_units": "Продажи, шт.",
	"total_sales_units": "Суммарные продажи, шт.",
	"mean_sales_units": "Средние продажи, шт.",
	"actual": "Факт",
	"prediction": "Прогноз",
	"absolute_error": "Абсолютная ошибка",
	"abs_error": "Абсолютная ошибка",
	"density": "Плотность",
	"count": "Количество",
	"number_of_series": "Число рядов",
	"item_count": "Число SKU",
	"history_weeks": "Длина истории, нед.",
	"zero_share": "Доля нулевых продаж",
	"missing_share": "Доля пропусков",
	"outlier_share": "Доля выбросов",
	"coefficient_of_variation": "Коэффициент вариации",
	"trend_strength": "R2 линейного тренда",
	"week_of_year": "Неделя года",
	"lag_weeks": "Лаг, нед.",
	"mean_acf": "Средняя автокорреляция",
	"seasonal_index": "Сезонный индекс",
	"seasonal_index_vs_category_mean": "Сезонный индекс к среднему категории",
	"sales / category_mean": "Продажи / среднее категории",
	"sales / sku_mean": "Продажи / среднее SKU",
	"cumulative_sales_share": "Накопленная доля продаж",
	"sku_rank": "Ранг SKU",
	"log1p(sales_units)": "log1p(продаж, шт.)",
	"WAPE": "WAPE",
	"sMAPE": "sMAPE",
	"Bias": "Смещение",
	"MAE": "MAE",
}


def translate_label(value: str) -> str:
	return PLOT_LABELS.get(value, value.replace("_", " "))


def set_axis_labels(
	ax: Axes,
	*,
	title: str | None = None,
	xlabel: str | None = None,
	ylabel: str | None = None,
) -> None:
	if title is not None:
		ax.set_title(title)
	if xlabel is not None:
		ax.set_xlabel(translate_label(xlabel))
	if ylabel is not None:
		ax.set_ylabel(translate_label(ylabel))
