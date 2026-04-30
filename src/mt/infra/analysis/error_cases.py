import pandas as pd


def select_error_cases(
	predictions: pd.DataFrame,
	baseline_model: str = "seasonal_naive",
	top_n: int = 20,
) -> pd.DataFrame:
	"""Выбрать диагностические кейсы выигрыша и проигрыша относительно baseline.

	Функция не пытается доказать превосходство модели статистически. Она нужна для
	качественного разбора, в каких рядах и на каких горизонтах модель заметно лучше
	или хуже обязательного baseline Seasonal Naive.
	"""

	if predictions.empty or baseline_model not in predictions["model_name"].unique():
		return pd.DataFrame()
	baseline = predictions[predictions["model_name"] == baseline_model].copy()
	baseline = baseline.rename(columns={"prediction": "baseline_prediction"})
	baseline["baseline_abs_error"] = (baseline["baseline_prediction"] - baseline["actual"]).abs()
	join_columns = ["series_id", "forecast_origin", "target_date", "horizon", "actual"]
	merged = predictions[predictions["model_name"] != baseline_model].merge(
		baseline[join_columns + ["baseline_prediction", "baseline_abs_error"]],
		on=join_columns,
		how="inner",
	)
	merged["model_abs_error"] = (merged["prediction"] - merged["actual"]).abs()
	merged["abs_error_gain_vs_baseline"] = merged["baseline_abs_error"] - merged["model_abs_error"]
	win_cases = merged.sort_values("abs_error_gain_vs_baseline", ascending=False).head(top_n)
	loss_cases = merged.sort_values("abs_error_gain_vs_baseline", ascending=True).head(top_n)
	selected = pd.concat(
		[win_cases.assign(case_type="best_win"), loss_cases.assign(case_type="worst_loss")],
		ignore_index=True)
	columns = [
		"case_type",
		"model_name",
		"series_id",
		"category",
		"segment_label",
		"forecast_origin",
		"target_date",
		"horizon",
		"actual",
		"prediction",
		"baseline_prediction",
		"model_abs_error",
		"baseline_abs_error",
		"abs_error_gain_vs_baseline",
	]
	return selected.loc[:, [column for column in columns if column in selected.columns]].reset_index(
		drop=True)
