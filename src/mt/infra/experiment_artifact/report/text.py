import pandas as pd

from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.domain.metric.metric_name import PointMetricName, ProbabilisticMetricName
from mt.domain.model.model_name import ModelName
from mt.infra.artifact.text_writer import write_markdown


def write_comparison_report(ctx: ExperimentPipelineContext) -> None:
	comparison_report = _build_comparison_report(
		ctx.overall_metrics,
		ctx.probabilistic_overall_metrics if ctx.probabilistic_overall_metrics is not None else pd.DataFrame(),
		ctx.evaluation.metrics_by_segment if ctx.evaluation is not None else pd.DataFrame(),
		ctx.evaluation.bootstrap_ci if ctx.evaluation is not None else pd.DataFrame(),
		ctx.evaluation.rolling_vs_holdout if ctx.evaluation is not None else pd.DataFrame(),
	)
	run_summary = _build_run_summary(
		ctx.overall_metrics,
		ctx.probabilistic_overall_metrics if ctx.probabilistic_overall_metrics is not None else pd.DataFrame(),
		ctx.model_feature_usage_rows,
		ctx.executed_stages,
		ctx.stage_timings,
	)
	write_markdown(
		ctx.artifacts_paths_map.report_file("REPORT.md"),
		_build_full_report(run_summary, comparison_report),
	)


def _build_run_summary(
	overall_metrics: pd.DataFrame,
	probabilistic_overall_metrics: pd.DataFrame,
	model_feature_usage_rows: list[dict[str, object]],
	executed_stages: list[str],
	stage_timings: list[dict[str, object]],
) -> list[str]:
	lines = ["## Этапы"]
	stage_timings_map = {str(stage_info["stage_name"]): stage_info for stage_info in stage_timings}
	for stage_name in executed_stages:
		stage_info = stage_timings_map.get(stage_name, {})
		lines.append(
			f"- {stage_name}: выполнено | wall_time={_format_seconds(stage_info.get('wall_time_seconds'))} сек."
		)

	lines.extend(["", "## Модели"])
	for row in model_feature_usage_rows:
		feature_status = (
			"tabular_features=used"
			if int(row["feature_count"]) > 0
			else "tabular_features=none"
		)
		lines.append(
			f"- {row['model_name']}: family={row['model_family']} | "
			f"feature_count={row['feature_count']} | "
			f"{feature_status}"
		)

	lines.extend(["", "## Что получилось"])
	lines.append("- Итоговая выбранная модель фиксируется в run/summary.yaml")
	lines.append("- Финальный сериализованный model artifact создается stage `experiment_best_model_training`")
	probabilistic_by_model = _build_probabilistic_metrics_by_model(probabilistic_overall_metrics)
	for rank, row in enumerate(overall_metrics.itertuples(index=False), start=1):
		lines.append(_build_model_result_line(rank, row, probabilistic_by_model.get(str(row.model_name), {})))

	return lines


def _build_comparison_report(
	overall_metrics: pd.DataFrame,
	probabilistic_overall_metrics: pd.DataFrame,
	metrics_by_segment: pd.DataFrame,
	bootstrap_ci: pd.DataFrame,
	rolling_vs_holdout: pd.DataFrame,
	baseline_name: ModelName = ModelName.SEASONAL_NAIVE,
) -> list[str]:
	lines = ["## Сравнение", "", "### Рейтинг"]
	if overall_metrics.empty:
		lines.append("- Сопоставимые результаты отсутствуют")
		return lines

	winner = overall_metrics.iloc[0]
	baseline_row = overall_metrics[overall_metrics["model_name"] == baseline_name]
	baseline_wape = float(
		baseline_row.iloc[0][PointMetricName.WAPE]) if not baseline_row.empty else None
	for row in overall_metrics.itertuples(index=False):
		delta_to_best = row.WAPE - winner[PointMetricName.WAPE]
		suffix = ""
		if baseline_wape is not None:
			suffix = f", разница_с_{baseline_name}={row.WAPE - baseline_wape:.4f}"
		lines.append(
			f"- {row.model_name}: WAPE={row.WAPE:.4f}, разница_с_лидером={delta_to_best:.4f}{suffix}"
		)
	if not probabilistic_overall_metrics.empty:
		lines.extend(["", "### Вероятностный рейтинг"])
		for row in probabilistic_overall_metrics.itertuples(index=False):
			if pd.isna(row.WIS):
				continue
			lines.append(
				f"- {row.model_name}: WIS={row.WIS:.4f}, MeanPinball={row.MeanPinball:.4f}, "
				f"Coverage80={row.Coverage80:.4f}, Coverage95={row.Coverage95:.4f}"
			)

	lines.extend(["", "### Устойчивость по сегментам"])
	best_segment = (
		metrics_by_segment.sort_values(["segment_label", PointMetricName.WAPE])
		.groupby("segment_label", as_index=False)
		.first()
	)
	for row in best_segment.itertuples(index=False):
		lines.append(f"- {row.segment_label}: best={row.model_name}, WAPE={row.WAPE:.4f}")

	lines.extend(["", "### Бутстрэп"])
	for row in bootstrap_ci.itertuples(index=False):
		lines.append(
			f"- {row.model_a} - {row.model_b}: средняя_разность={row.mean_difference:.4f}, "
			f"95% ДИ=[{row.ci95_low:.4f}, {row.ci95_high:.4f}]"
		)

	lines.extend(["", "### Rolling Vs Holdout"])
	multi_origin = bool(
		(rolling_vs_holdout["variance_reduction_factor_vs_single_holdout"] > 1).any())
	if multi_origin:
		lines.append(
			"- При допущении конечной дисперсии потерь по forecast origin средняя rolling-оценка по K окнам имеет дисперсию порядка sigma^2 / K, тогда как одиночный holdout соответствует K = 1."
		)
	else:
		lines.append(
			"- Для текущего контура доступен только один forecast origin, поэтому rolling и last-origin holdout совпадают и оценка устойчивости ограничена."
		)
	for row in rolling_vs_holdout.itertuples(index=False):
		lines.append(
			f"- {row.model_name}: rolling_WAPE={row.rolling_WAPE:.4f}, "
			f"holdout_last_origin={row.holdout_WAPE_last_origin:.4f}, "
			f"std_WAPE_по_origin={_format_metric_or_na(row.origin_WAPE_std)}, "
			f"SE_rolling={_format_metric_or_na(row.rolling_WAPE_standard_error)}, "
			f"Фактор_снижения_дисперсии~{row.variance_reduction_factor_vs_single_holdout:.1f}"
		)

	lines.extend(["", "### Финальная рекомендация"])
	lines.append(f"- Выбранная модель: {winner['model_name']}")
	lines.append(f"- Метрика выбора: WAPE={winner[PointMetricName.WAPE]:.4f}")

	if baseline_wape is not None:
		lines.append(
			f"- Улучшение_относительно_{baseline_name}: "
			f"{baseline_wape - winner[PointMetricName.WAPE]:.4f}"
		)

	return lines


def _build_full_report(run_summary: list[str], comparison_report: list[str]) -> list[str]:
	lines = [
		"# Experiment REPORT",
		"",
	]
	lines.extend(run_summary)
	lines.extend(["", ""])
	lines.extend(comparison_report)
	return lines


def _build_probabilistic_metrics_by_model(frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
	if frame.empty or "model_name" not in frame.columns:
		return {}
	payload: dict[str, dict[str, float | None]] = {}
	for _, row in frame.iterrows():
		payload[str(row["model_name"])] = {
			str(metric_name): _coerce_metric_value(row.get(str(metric_name)))
			for metric_name in ProbabilisticMetricName
		}
	return payload


def _build_model_result_line(
	rank: int,
	row: tuple[object, ...],
	probabilistic_metrics: dict[str, float | None],
) -> str:
	point_parts = [
		f"WAPE={_format_numeric_value(getattr(row, 'WAPE', None))}",
		f"sMAPE={_format_numeric_value(getattr(row, 'sMAPE', None))}",
		f"MAE={_format_numeric_value(getattr(row, 'MAE', None))}",
		f"RMSE={_format_numeric_value(getattr(row, 'RMSE', None))}",
		f"Bias={_format_numeric_value(getattr(row, 'Bias', None))}",
		f"MedianAE={_format_numeric_value(getattr(row, 'MedianAE', None))}",
	]
	prob_parts = [
		f"{metric_name}={_format_numeric_value(probabilistic_metrics.get(str(metric_name)))}"
		for metric_name in ProbabilisticMetricName
	]
	return f"- место {rank}: {row.model_name} | " + " | ".join([*point_parts, *prob_parts])


def _format_seconds(value: float | None) -> str:
	if value is None:
		return "n/a"
	return f"{value:.3f}"


def _format_metric_or_na(value: float | None) -> str:
	if value is None or pd.isna(value):
		return "n/a"
	return f"{value:.4f}"


def _coerce_metric_value(value: object) -> float | None:
	if value is None or pd.isna(value):
		return None
	return float(value)


def _format_numeric_value(value: object) -> str:
	if value is None or pd.isna(value):
		return "n/a"
	return f"{float(value):.4f}"
