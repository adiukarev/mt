from dataclasses import dataclass

import pandas as pd

from mt.infra.metric.calculates import calculate_metrics
from .bootstrap import build_bootstrap_ci
from .error_cases import select_error_cases


@dataclass(slots=True)
class ComparisonArtifacts:
	"""Артефакты итогового сравнения моделей"""

	metrics_by_segment: pd.DataFrame
	metrics_by_category: pd.DataFrame
	bootstrap_ci: pd.DataFrame
	error_cases: pd.DataFrame
	rolling_vs_holdout: pd.DataFrame


def summarize_by_dimension(predictions: pd.DataFrame, dimension: str) -> pd.DataFrame:
	"""Агрегировать метрики по модели и дополнительному измерению"""

	if predictions.empty or dimension not in predictions.columns:
		return pd.DataFrame()
	rows: list[dict[str, object]] = []
	for (model_name, value), frame in predictions.groupby(["model_name", dimension]):
		if pd.isna(value):
			continue
		rows.append({"model_name": model_name, dimension: value, **calculate_metrics(frame)})
	return pd.DataFrame(rows).sort_values(["model_name", dimension]).reset_index(drop=True)


def build_comparison_artifacts(
	predictions: pd.DataFrame,
	seed: int,
	bootstrap_samples: int = 300,
) -> ComparisonArtifacts:
	"""Собрать артефакты итогового сравнения после rolling backtesting.

	Здесь формируются не только агрегированные метрики, но и диагностические
	артефакты, которые помогают интерпретировать устойчивость результата:
	сравнение по сегментам, по категориям, bootstrap CI и last-origin holdout
	как дополнительная проверка хрупкости вывода.
	"""

	metrics_by_segment = summarize_by_dimension(predictions, "segment_label")
	metrics_by_category = summarize_by_dimension(predictions, "category")
	bootstrap_ci = build_bootstrap_ci(
		predictions,
		seed=seed,
		n_bootstrap=bootstrap_samples,
	)
	error_cases = select_error_cases(predictions)
	rolling_vs_holdout = build_rolling_holdout_diagnostic(predictions)

	return ComparisonArtifacts(
		metrics_by_segment=metrics_by_segment,
		metrics_by_category=metrics_by_category,
		bootstrap_ci=bootstrap_ci,
		error_cases=error_cases,
		rolling_vs_holdout=rolling_vs_holdout,
	)


def build_rolling_holdout_diagnostic(predictions: pd.DataFrame) -> pd.DataFrame:
	"""Сравнить rolling-оценку с одиночным holdout по последнему forecast origin"""

	if predictions.empty:
		return pd.DataFrame()

	last_origin = predictions["forecast_origin"].max()
	holdout_predictions = predictions[predictions["forecast_origin"] == last_origin].copy()
	rows: list[dict[str, object]] = []
	for model_name, model_frame in predictions.groupby("model_name"):
		rows.append(
			_build_rolling_holdout_row(
				model_name=model_name,
				model_frame=model_frame,
				holdout_predictions=holdout_predictions,
			)
		)

	return pd.DataFrame(rows).sort_values("rolling_WAPE").reset_index(drop=True)


def _build_rolling_holdout_row(
	model_name: str,
	model_frame: pd.DataFrame,
	holdout_predictions: pd.DataFrame,
) -> dict[str, object]:
	"""Собрать диагностическую строку по одной модели"""

	per_origin_wape = _summarize_origin_wape(model_frame)
	origin_count = int(len(per_origin_wape))
	origin_wape_std = float(per_origin_wape["WAPE"].std(ddof=1)) if origin_count > 1 else float("nan")
	rolling_standard_error = (
		float(origin_wape_std / origin_count ** 0.5)
		if origin_count > 1 and pd.notna(origin_wape_std)
		else float("nan")
	)
	rolling_metrics = calculate_metrics(model_frame)
	holdout_frame = holdout_predictions[holdout_predictions["model_name"] == model_name]
	holdout_wape = (
		calculate_metrics(holdout_frame)["WAPE"]
		if not holdout_frame.empty
		else float("nan")
	)

	return {
		"model_name": model_name,
		"origins_count": origin_count,
		"rolling_WAPE": rolling_metrics["WAPE"],
		"holdout_WAPE_last_origin": holdout_wape,
		"holdout_minus_rolling_WAPE": holdout_wape - rolling_metrics["WAPE"],
		"origin_WAPE_std": origin_wape_std,
		"rolling_WAPE_standard_error": rolling_standard_error,
		"variance_reduction_factor_vs_single_holdout": float(
			origin_count) if origin_count > 0 else float("nan"),
	}


def _summarize_origin_wape(model_frame: pd.DataFrame) -> pd.DataFrame:
	"""Посчитать WAPE по каждому forecast origin для одной модели."""

	return pd.DataFrame(
		[
			{
				"forecast_origin": forecast_origin,
				"WAPE": calculate_metrics(origin_frame)["WAPE"],
			}
			for forecast_origin, origin_frame in model_frame.groupby("forecast_origin")
		]
	)
