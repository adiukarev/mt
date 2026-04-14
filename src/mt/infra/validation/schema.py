import pandas as pd


def validate_dataset_schema(
	weekly: pd.DataFrame,
	aggregation_level: str
) -> dict[str, object]:
	"""Провалидировать схему датасета на обязательные колонки и целостность ключей после недельной агрегации"""

	required_columns = {"series_id", "category", "week_start", "sales_units"}
	missing_columns = sorted(required_columns.difference(weekly.columns))
	duplicate_rows = int(
		weekly.duplicated(subset=["series_id", "week_start"]).sum()) if not missing_columns else -1
	non_datetime_week = "week_start" in weekly.columns and not pd.api.types.is_datetime64_any_dtype(
		weekly["week_start"]
	)
	non_monotonic = False

	if not missing_columns:
		non_monotonic = bool(
			weekly.sort_values(["series_id", "week_start"]).groupby("series_id")[
				"week_start"].diff().dropna().lt(pd.Timedelta(0)).any()
		)

	summary = {
		"aggregation_level": aggregation_level,
		"required_columns": sorted(required_columns),
		"missing_columns": missing_columns,
		"duplicate_rows_after_aggregation": duplicate_rows,
		"week_start_is_datetime": not non_datetime_week,
		"time_axis_monotonic": not non_monotonic,
		"schema_ok": not missing_columns and duplicate_rows == 0 and not non_datetime_week and not non_monotonic,
	}
	if not summary["schema_ok"]:
		raise ValueError()
	return summary
