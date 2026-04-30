from dataclasses import asdict
from dataclasses import dataclass

import pandas as pd

from mt.domain.feature.feature_manifest import FeatureManifest
from mt.domain.feature.feature_history_formula_catalog import (
	build_lag_feature_specs,
	build_rolling_feature_specs,
)
from mt.domain.feature.feature_set import FeatureGroup, get_feature_groups


@dataclass(slots=True)
class FeatureSpec:
	"""Внутренняя спецификация признака для feature registry"""

	name: str
	group: str
	source: str
	calculation: str
	covariate_class: str
	expected_effect_mechanism: str
	availability_at_forecast_time: bool
	enabled: bool = True
	reason_if_disabled: str = ""


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
		get_feature_groups(manifest.feature_set)
		if manifest.enabled
		else frozenset()
	)
	specs: list[FeatureSpec] = []

	for lag_spec in build_lag_feature_specs(manifest.lags):
		specs.append(
			FeatureSpec(
				name=lag_spec.name,
				group=FeatureGroup.LAGS.value,
				source="исторические продажи",
				calculation=lag_spec.formula,
				covariate_class="observed",
				expected_effect_mechanism=(
					"Улавливает инерцию, локальную устойчивость и повторяющуюся "
					"недельную или годовую память по уже наблюденным продажам."
				),
				availability_at_forecast_time=True,
				enabled=FeatureGroup.LAGS in enabled_groups,
				reason_if_disabled="" if FeatureGroup.LAGS in enabled_groups else "набор признаков не включает лаги",
			)
		)

	for rolling_spec in build_rolling_feature_specs(manifest.rolling_windows):
		specs.append(
			FeatureSpec(
				name=rolling_spec.name,
				group=FeatureGroup.ROLLING.value,
				source="исторические продажи",
				calculation=rolling_spec.formula,
				covariate_class="observed",
				expected_effect_mechanism=(
					"Приближает локальный уровень, волатильность и краткосрочный "
					"тренд, используя только прошлую историю продаж."
				),
				availability_at_forecast_time=True,
				enabled=FeatureGroup.ROLLING in enabled_groups,
				reason_if_disabled="" if FeatureGroup.ROLLING in enabled_groups else "набор признаков не включает rolling-статистики",
			)
		)

	for window in manifest.rolling_windows:
		specs.extend(
			[
				FeatureSpec(
					name=f"rolling_ratio_to_mean_{window}",
					group=FeatureGroup.ROLLING.value,
					source="исторические продажи",
					calculation=(
						f"rolling_ratio_to_mean_{window}(t) = y_(t-1) / rolling_mean_{window}(t)"
					),
					covariate_class="observed",
					expected_effect_mechanism=(
						"Сравнивает последнее наблюдение с локальным уровнем и помогает "
						"заметить краткосрочные всплески или провалы."
					),
					availability_at_forecast_time=True,
					enabled=FeatureGroup.ROLLING in enabled_groups,
					reason_if_disabled="" if FeatureGroup.ROLLING in enabled_groups else "набор признаков не включает rolling-статистики",
				),
				FeatureSpec(
					name=f"rolling_robust_zscore_{window}",
					group=FeatureGroup.ROLLING.value,
					source="исторические продажи",
					calculation=(
						f"rolling_robust_zscore_{window}(t) = 0.6745 * "
						f"(y_(t-1) - rolling_median_{window}(t)) / rolling_mad_{window}(t)"
					),
					covariate_class="observed",
					expected_effect_mechanism=(
						"Показывает, насколько последнее наблюдение отклонилось от "
						"локального устойчивого уровня с поправкой на robust-разброс."
					),
					availability_at_forecast_time=True,
					enabled=FeatureGroup.ROLLING in enabled_groups,
					reason_if_disabled="" if FeatureGroup.ROLLING in enabled_groups else "набор признаков не включает rolling-статистики",
				),
				FeatureSpec(
					name=f"rolling_recent_outlier_flag_{window}",
					group=FeatureGroup.ROLLING.value,
					source="исторические продажи",
					calculation=(
						f"rolling_recent_outlier_flag_{window}(t) = "
						f"|rolling_robust_zscore_{window}(t)| > 3.5"
					),
					covariate_class="observed",
					expected_effect_mechanism=(
						"Маркирует, что последнее наблюдение похоже на локальный spike "
						"или провал относительно устойчивой истории окна."
					),
					availability_at_forecast_time=True,
					enabled=FeatureGroup.ROLLING in enabled_groups,
					reason_if_disabled="" if FeatureGroup.ROLLING in enabled_groups else "набор признаков не включает rolling-статистики",
				),
				FeatureSpec(
					name=f"rolling_recent_trend_delta_{window}",
					group=FeatureGroup.ROLLING.value,
					source="исторические продажи",
					calculation=(
						f"rolling_recent_trend_delta_{window}(t) = y_(t-1) - y_(t-{window})"
					),
					covariate_class="observed",
					expected_effect_mechanism=(
						"Приближает недавний сдвиг уровня относительно начала окна."
					),
					availability_at_forecast_time=True,
					enabled=FeatureGroup.ROLLING in enabled_groups,
					reason_if_disabled="" if FeatureGroup.ROLLING in enabled_groups else "набор признаков не включает rolling-статистики",
				),
			]
		)

	for name in ("week_of_year", "month", "quarter", "year", "week_in_month"):
		specs.append(
			FeatureSpec(
				name=name,
				group=FeatureGroup.CALENDAR.value,
				source="календарь",
				calculation=f"непосредственно вычисляется из week_start ({name})",
				covariate_class="known_in_advance",
				expected_effect_mechanism=(
					"Кодирует детерминированное сезонное положение в retail-календаре, "
					"которое известно заранее до даты прогноза."
				),
				availability_at_forecast_time=True,
				enabled=FeatureGroup.CALENDAR in enabled_groups and manifest.use_calendar,
				reason_if_disabled="" if FeatureGroup.CALENDAR in enabled_groups and manifest.use_calendar else "календарные признаки отключены",
			)
		)

	categorical_specs: list[tuple[str, bool]] = [
		("category_code", True),
		("segment_code", True),
		("sku_code", aggregation_level == "sku"),
	]
	for name, available_for_level in categorical_specs:
		enabled = (
			FeatureGroup.CATEGORICAL in enabled_groups
			and manifest.use_category_encodings
			and available_for_level
		)
		specs.append(
			FeatureSpec(
				name=name,
				group=FeatureGroup.CATEGORICAL.value,
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
					if FeatureGroup.CATEGORICAL not in enabled_groups or not manifest.use_category_encodings
					else "признак недоступен на текущем уровне агрегации"
				),
			)
		)

	return pd.DataFrame(asdict(spec) for spec in specs)
