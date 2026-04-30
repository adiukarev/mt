from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuditReportSpec:
	key: str
	label: str
	variables: tuple[str, ...]
	interpretation: str
	model_action: str
	feature_formula_policy: str


AUDIT_REPORT_SPECS: tuple[AuditReportSpec, ...] = (
	AuditReportSpec(
		key="series_identity",
		label="Ряд",
		variables=("series_id", "category", "history_weeks"),
		interpretation="Фиксирует конкретный ряд, уровень агрегации и длину доступной истории именно для этого сценария.",
		model_action="Проверить, что row-level benchmark не подменяет multi-series выводы.",
		feature_formula_policy="Структурные признаки задаются через `series_id`, `category`, `segment_label`.",
	),
	AuditReportSpec(
		key="sales_magnitude",
		label="Масштаб",
		variables=("total_sales_units", "mean_sales_units"),
		interpretation="Показывает суммарный и средний уровень продаж ряда; это влияет на шум, относительные ошибки и полезность нормализации.",
		model_action="Сверить loss и baseline с масштабом ряда.",
		feature_formula_policy="Для нестабильного масштаба полезны `rolling_ratio_to_mean_7` и стандартизованные rolling-признаки.",
	),
	AuditReportSpec(
		key="sales_scale",
		label="Масштаб продаж",
		variables=("sales_scale",),
		interpretation="Показывает типичный абсолютный уровень ряда в исходных единицах.",
		model_action="Основной metric focus лучше держать на WAPE/MAE, а не только на относительных метриках.",
		feature_formula_policy="Формулы лагов не меняются; при необходимости downstream scaling делается отдельно.",
	),
	AuditReportSpec(
		key="missingness",
		label="Пропуски",
		variables=("missing_share",),
		interpretation="Показывает, насколько полна недельная сетка именно этого ряда.",
		model_action="Перед моделированием явно валидировать weekly grid для ряда.",
		feature_formula_policy="Лаги и rolling считать только на согласованной недельной панели.",
	),
	AuditReportSpec(
		key="distribution",
		label="Распределение",
		variables=("skewness",),
		interpretation="Скошенность показывает, насколько у ряда тяжелый правый хвост и редкие пики.",
		model_action="При сильной асимметрии использовать robust-loss и медианные агрегаты.",
		feature_formula_policy="Опираться на `rolling_median_7/28`, а не только на средние.",
	),
	AuditReportSpec(
		key="zeros",
		label="Нули",
		variables=("zero_share",),
		interpretation="Описывает разреженность спроса на уровне конкретного ряда.",
		model_action="Проверить, нужен ли intermittent-aware подход.",
		feature_formula_policy="При высоком `zero_share` добавить zero-run / demand-memory признаки и рассматривать robust-loss.",
	),
	AuditReportSpec(
		key="noise",
		label="Шум",
		variables=("coefficient_of_variation", "volatility"),
		interpretation="Показывает относительную и абсолютную нестабильность ряда.",
		model_action="Чем выше шум, тем осторожнее нужно выбирать сложные модели и loss.",
		feature_formula_policy="Усилять `rolling_std`, медианные rolling-окна и robust-признаки при высоком шуме.",
	),
	AuditReportSpec(
		key="outliers",
		label="Выбросы",
		variables=("outlier_share",),
		interpretation="Показывает, насколько часто в ряду есть аномальные пики или провалы.",
		model_action="Не удалять пики без отдельного обоснования.",
		feature_formula_policy="Опираться на robust rolling statistics; при низком `outlier_share` обычно достаточно `rolling_mean`, `rolling_std`, `rolling_max/min`.",
	),
	AuditReportSpec(
		key="mean_shift",
		label="Сдвиг уровня",
		variables=("mean_shift_ratio", "trend_strength"),
		interpretation="Показывает, насколько изменился средний уровень и какую долю вариации объясняет линейный тренд по времени.",
		model_action="Если сдвиг или тренд заметны, усиливать trend-aware блок модели.",
		feature_formula_policy="При заметном сдвиге добавить `time_idx`, rolling trend delta и при необходимости diff-признаки.",
	),
	AuditReportSpec(
		key="variance_shift",
		label="Сдвиг дисперсии",
		variables=("variance_shift_ratio",),
		interpretation="Показывает, насколько изменился разброс ряда между частями истории.",
		model_action="Сверить устойчивость модели к росту шума.",
		feature_formula_policy="При росте дисперсии использовать robust scaling, rolling std и медианные агрегаты.",
	),
	AuditReportSpec(
		key="short_memory",
		label="Автокорреляция",
		variables=("acf_lag_1", "acf_lag_7"),
		interpretation="Показывает краткосрочную память ряда и полезность lag-based постановки.",
		model_action="Если автокорреляция слабая, нельзя рассчитывать только на короткие лаги.",
		feature_formula_policy="Базовый memory-block: `lag_1`, `lag_7`, `rolling_mean_7`, `rolling_median_7`.",
	),
	AuditReportSpec(
		key="seasonality",
		label="Сезонность",
		variables=("acf_lag_28", "acf_lag_52"),
		interpretation="Показывает силу среднесрочной и годовой сезонной памяти для данного ряда.",
		model_action="Обязательно сравнивать с `Seasonal Naive`, если сезонность заметна.",
		feature_formula_policy="При сильном `acf_lag_52` добавить `lag_52`, `rolling_mean_52`, seasonal baseline и WOY-календарь; при заметном `acf_lag_28` усилить `lag_28` и окна 28 недель.",
	),
	AuditReportSpec(
		key="dependency_focus",
		label="Зависимости",
		variables=("acf_focus_lags",),
		interpretation="Эвристически фиксирует, на каких лагах audit видит наиболее полезную историческую память именно для этого ряда.",
		model_action="От этих lag-узлов нужно строить supervised feature block и baseline selection.",
		feature_formula_policy="Это heuristic feature-policy, а не тест статистической значимости: выбирать lag/rolling окна вокруг реально сильных lag-точек, а не по одному общему шаблону.",
	),
	AuditReportSpec(
		key="stationarity",
		label="Стационарность",
		variables=("adf_pvalue", "kpss_pvalue", "kpss_pvalue_note", "stationarity_hint"),
		interpretation="Показывает, насколько ряд похож на стационарный по ADF/KPSS-подсказке, и отдельно фиксирует, когда KPSS p-value упёрся в границу таблицы.",
		model_action="Сверить необходимость differencing и trend-aware моделей.",
		feature_formula_policy="При `likely_non_stationary*` добавить diff/ratio features и держать тренд-aware модели; `kpss_pvalue_note` нужно читать как качество evidence, а не как feature.",
	),
)
