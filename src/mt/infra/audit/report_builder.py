import pandas as pd

from mt.domain.audit.audit_report_spec import AUDIT_REPORT_SPECS


def build_report_lines(
	series_diagnostic_table: pd.DataFrame,
) -> list[str]:
	lines = [
		"# Audit Report",
	]
	if not series_diagnostic_table.empty:
		selected_series_table = build_selected_series_table(series_diagnostic_table)
		lines.extend(
			[
				"",
				"## Selected Series",
				"",
			]
		)
		lines.extend(_render_markdown_table(selected_series_table))
		lines.extend(
			[
				"",
				"## Analysis Blocks",
				"",
				"| Блок анализа | Переменные | Что это значит | Что делать в модели | Feature Formula Policy |",
				"| --- | --- | --- | --- | --- |",
			]
		)
		for diagnostic_row in _build_diagnostic_reference_rows():
			lines.append(
				f"| {diagnostic_row['analysis_block']} | {diagnostic_row['variables']} | {diagnostic_row['value']} | {diagnostic_row['what_to_do_in_model']} | {diagnostic_row['feature_formula_policy']} |"
			)
	return lines


def build_series_diagnostic_table(
	summary: pd.DataFrame,
	seasonality_summary: pd.DataFrame,
	stationarity_summary: pd.DataFrame,
) -> pd.DataFrame:
	seasonality_columns = [
		column for column in ("series_id", "acf_lag_1", "acf_lag_7", "acf_lag_28", "acf_lag_52")
		if column in seasonality_summary.columns
	]
	stationarity_columns = [
		column for column in ("series_id", "adf_pvalue", "kpss_pvalue", "kpss_pvalue_note", "stationarity_hint")
		if column in stationarity_summary.columns
	]
	result = summary.copy()
	if seasonality_columns:
		result = result.merge(
			seasonality_summary.loc[:, seasonality_columns],
			on="series_id",
			how="left",
		)
	if stationarity_columns:
		result = result.merge(
			stationarity_summary.loc[:, stationarity_columns],
			on="series_id",
			how="left",
		)

	ordered_columns = [
		column for column in (
			"series_id",
			"category",
			"segment_label",
			"history_weeks",
			"total_sales_units",
			"mean_sales_units",
			"zero_share",
			"missing_share",
			"outlier_share",
			"coefficient_of_variation",
			"skewness",
			"sales_scale",
			"volatility",
			"trend_strength",
			"mean_shift_ratio",
			"variance_shift_ratio",
			"acf_lag_1",
			"acf_lag_7",
			"acf_lag_28",
			"acf_lag_52",
			"acf_focus_lags",
			"adf_pvalue",
			"kpss_pvalue",
			"kpss_pvalue_note",
			"stationarity_hint",
		)
		if column in result.columns
	]
	return result.loc[:, ordered_columns].sort_values("series_id").reset_index(drop=True)


def build_selected_series_table(series_diagnostic_table: pd.DataFrame) -> pd.DataFrame:
	return series_diagnostic_table.rename(columns=_display_label_for_column).reset_index(drop=True)


def _render_markdown_table(frame: pd.DataFrame) -> list[str]:
	if frame.empty:
		return []
	rendered = frame.copy()
	for column in rendered.columns:
		series = rendered[column]
		if pd.api.types.is_float_dtype(series):
			rendered[column] = series.map(_format_float)
		else:
			rendered[column] = series.fillna("").astype(str)

	header = "| " + " | ".join(rendered.columns.tolist()) + " |"
	separator = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
	rows = [
		"| " + " | ".join(row) + " |"
		for row in rendered.astype(str).itertuples(index=False, name=None)
	]
	return [header, separator, *rows]


def _format_float(value: object) -> str:
	if value is None or pd.isna(value):
		return ""
	return f"{float(value):.4f}"


def _build_diagnostic_reference_rows() -> list[dict[str, str]]:
	return [
		_reference_row(
			spec.label,
			_format_variable_list(spec.variables),
			spec.interpretation,
			spec.model_action,
			spec.feature_formula_policy,
		)
		for spec in AUDIT_REPORT_SPECS
	]


def _reference_row(
	analysis_block: str,
	variables: str,
	interpretation: str,
	model_action: str,
	feature_formula_policy: str,
) -> dict[str, str]:
	return {
		"analysis_block": analysis_block,
		"variables": variables,
		"value": interpretation,
		"what_to_do_in_model": model_action,
		"feature_formula_policy": feature_formula_policy,
	}


def _format_variable_list(variables: tuple[str, ...]) -> str:
	return "<br>".join(_display_label_for_variable(variable) for variable in variables)


def _display_label_for_column(column: str) -> str:
	labels = {
		"history_weeks": "history_weeks (нед.)",
		"total_sales_units": "total_sales_units (шт.)",
		"mean_sales_units": "mean_sales_units (шт./нед.)",
		"sales_scale": "sales_scale (шт./нед.)",
		"volatility": "volatility (шт./нед.)",
		"zero_share": "zero_share (доля)",
		"missing_share": "missing_share (доля)",
		"outlier_share": "outlier_share (доля)",
		"coefficient_of_variation": "coefficient_of_variation (доля)",
		"trend_strength": "trend_strength (R2)",
		"mean_shift_ratio": "mean_shift_ratio (отношение)",
		"variance_shift_ratio": "variance_shift_ratio (отношение)",
		"acf_lag_1": "acf_lag_1 (corr)",
		"acf_lag_7": "acf_lag_7 (corr)",
		"acf_lag_28": "acf_lag_28 (corr)",
		"acf_lag_52": "acf_lag_52 (corr)",
		"acf_focus_lags": "acf_focus_lags (нед.)",
		"adf_pvalue": "adf_pvalue (p-value)",
		"kpss_pvalue": "kpss_pvalue (p-value)",
	}
	return labels.get(column, column)


def _display_label_for_variable(variable: str) -> str:
	return _display_label_for_column(variable)
