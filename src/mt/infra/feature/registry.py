from dataclasses import asdict

import pandas as pd

from mt.domain.manifest import FeatureManifest
from mt.domain.feature import FeatureSpec

FEATURE_SET_GROUPS: dict[str, set[str]] = {
	"F0": set(),
	"F1": {"lags"},
	"F2": {"lags", "rolling"},
	"F3": {"lags", "rolling", "calendar"},
	"F4": {"lags", "rolling", "calendar", "categorical"},
	"F5": {"lags", "rolling", "calendar", "categorical", "price_promo"},
	"F6": {"lags", "rolling", "calendar", "categorical", "price_promo", "external"},
}
FEATURE_SET_ALIASES: dict[str, str] = {
	"F4_smoke": "F4",
}


def build_feature_registry(
	manifest: FeatureManifest,
	aggregation_level: str = "category",
) -> pd.DataFrame:
	"""Создать реестр признаков с явной политикой доступности.

	Реестр нужен не только для генерации колонок, но и как формальный документ:
	из чего считается признак, к какому классу ковариат относится и почему
	его можно или нельзя использовать на дату прогноза.
	"""

	enabled_groups = (
		FEATURE_SET_GROUPS[resolve_feature_set_groups_key(manifest.feature_set)]
		if manifest.enabled
		else set()
	)
	specs: list[FeatureSpec] = []

	for lag in manifest.lags:
		specs.append(
			FeatureSpec(
				name=f"lag_{lag}",
				group="lags",
				source="исторические продажи",
				calculation=f"sales_units, сдвинутые в прошлое на {lag} недель",
				covariate_class="observed",
				expected_effect_mechanism=(
					"Улавливает инерцию, локальную устойчивость и повторяющуюся "
					"недельную или годовую память по уже наблюденным продажам."
				),
				availability_at_forecast_time=True,
				enabled="lags" in enabled_groups,
				reason_if_disabled="" if "lags" in enabled_groups else "набор признаков не включает лаги",
			)
		)

	for window in manifest.rolling_windows:
		for metric in ("mean", "median", "std", "min", "max", "ratio_to_rolling_mean",
		               "recent_trend_delta"):
			specs.append(
				FeatureSpec(
					name=f"rolling_{window}_{metric}",
					group="rolling",
					source="исторические продажи",
					calculation=(
						f"sales_units со сдвигом на 1 неделю и агрегацией по "
						f"скользящему окну длиной {window} недель"
					),
					covariate_class="observed",
					expected_effect_mechanism=(
						"Приближает локальный уровень, волатильность и краткосрочный "
						"тренд, используя только прошлую историю продаж."
					),
					availability_at_forecast_time=True,
					enabled="rolling" in enabled_groups,
					reason_if_disabled="" if "rolling" in enabled_groups else "набор признаков не включает rolling-статистики",
				)
			)

	for name in ("week_of_year", "month", "quarter", "year", "week_in_month"):
		specs.append(
			FeatureSpec(
				name=name,
				group="calendar",
				source="календарь",
				calculation=f"непосредственно вычисляется из week_start ({name})",
				covariate_class="known_in_advance",
				expected_effect_mechanism=(
					"Кодирует детерминированное сезонное положение в retail-календаре, "
					"которое известно заранее до даты прогноза."
				),
				availability_at_forecast_time=True,
				enabled="calendar" in enabled_groups and manifest.use_calendar,
				reason_if_disabled="" if "calendar" in enabled_groups and manifest.use_calendar else "календарные признаки отключены",
			)
		)

	categorical_specs: list[tuple[str, bool]] = [
		("category_code", True),
		("segment_code", True),
		("sku_code", aggregation_level == "sku"),
	]
	for name, available_for_level in categorical_specs:
		enabled = "categorical" in enabled_groups and manifest.use_category_encodings and available_for_level
		specs.append(
			FeatureSpec(
				name=name,
				group="categorical",
				source="метаданные ряда",
				calculation=f"категориальное кодирование для {name}",
				covariate_class="known_in_advance",
				expected_effect_mechanism=(
					"Переносит устойчивые различия между рядами, например профиль "
					"категории или сегмент спроса."
				),
				availability_at_forecast_time=True,
				enabled=enabled,
				reason_if_disabled="" if enabled else (
					"категориальные признаки отключены"
					if "categorical" not in enabled_groups or not manifest.use_category_encodings
					else "признак недоступен на текущем уровне агрегации"
				),
			)
		)

	for name, enabled_flag in (("price", manifest.use_price), ("promo", manifest.use_promo)):
		specs.append(
			FeatureSpec(
				name=name,
				group="price_promo",
				source="retail-ковариаты",
				calculation=f"прямая ковариата '{name}' из операционных данных",
				covariate_class="future_unknown",
				expected_effect_mechanism=(
					"Описывает управляемые драйверы спроса, но прямое использование "
					"запрещено, пока доступность на дату прогноза не доказана отдельно."
				),
				availability_at_forecast_time=enabled_flag,
				enabled="price_promo" in enabled_groups and enabled_flag,
				reason_if_disabled="" if "price_promo" in enabled_groups and enabled_flag else "доступность на дату прогноза не доказана",
			)
		)

	specs.append(
		FeatureSpec(
			name="external_signal",
			group="external",
			source="внешний источник",
			calculation="внешняя ковариата, опубликованная до даты прогноза",
			covariate_class="future_unknown",
			expected_effect_mechanism=(
				"Может объяснять внешние сдвиги спроса только тогда, когда "
				"документированы лаг публикации и доступность в будущее."
			),
			availability_at_forecast_time=manifest.use_external,
			enabled="external" in enabled_groups and manifest.use_external,
			reason_if_disabled="" if "external" in enabled_groups and manifest.use_external else "внешние данные отключены",
		)
	)

	return pd.DataFrame(asdict(spec) for spec in specs)


def resolve_feature_set_groups_key(feature_set: str) -> str:
	"""Сопоставить исследовательский alias с канонической группой признаков"""

	if feature_set in FEATURE_SET_GROUPS:
		return feature_set

	canonical_feature_set = FEATURE_SET_ALIASES.get(feature_set)
	if canonical_feature_set is not None:
		return canonical_feature_set

	prefix = feature_set.split("_", maxsplit=1)[0]
	if prefix in FEATURE_SET_GROUPS:
		return prefix

	return "F4"
