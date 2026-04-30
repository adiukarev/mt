import pandas as pd

from mt.domain.feature.feature_manifest import FeatureManifest
from mt.domain.feature.feature_history_formula_default import DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS


def build_data_dictionary(weekly: pd.DataFrame) -> pd.DataFrame:
	descriptions = {
		"series_id": "Идентификатор временного ряда на выбранном уровне агрегации",
		"category": "Категория или surrogate business grouping, сформированный адаптером датасета",
		"week_start": "Дата начала недели с якорем W-MON",
		"sales_units": "Канонический downstream target. Для некоторых датасетов это может быть инженерный proxy, зафиксированный в metadata.",
	}
	role_map = {
		"series_id": "key",
		"category": "static",
		"week_start": "time_index",
		"sales_units": "target",
	}
	rows: list[dict[str, object]] = []
	for column in weekly.columns:
		rows.append(
			{
				"column_name": column,
				"dtype": str(weekly[column].dtype),
				"role": role_map.get(column, "derived"),
				"non_null_share": float(weekly[column].notna().mean()),
				"example_value": str(weekly[column].dropna().iloc[0]) if weekly[column].notna().any() else "",
				"description": descriptions.get(column),
			}
		)
	return pd.DataFrame(rows)


def default_feature_manifest() -> FeatureManifest:
	return FeatureManifest(
		enabled=True,
		feature_set="F4",
		lags=list(DEFAULT_LAGS),
		rolling_windows=list(DEFAULT_ROLLING_WINDOWS),
		use_calendar=True,
		use_category_encodings=True,
	)
