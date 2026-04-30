import pandas as pd

from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.infra.artifact.text_writer import write_csv


def write_overall_model_evaluation(ctx: ExperimentPipelineContext) -> None:
	if ctx.overall_metrics is None or ctx.overall_metrics.empty:
		return
	write_csv(
		ctx.artifacts_paths_map.evaluation_file("overall_model_evaluation.csv"),
		ctx.overall_metrics,
	)


def write_metrics_by_horizon_evaluation(ctx: ExperimentPipelineContext) -> None:
	if ctx.by_horizon_metrics is None or ctx.by_horizon_metrics.empty:
		return
	write_csv(
		ctx.artifacts_paths_map.evaluation_file("metrics_by_horizon.csv"),
		ctx.by_horizon_metrics,
	)


def write_probabilistic_overall_model_evaluation(ctx: ExperimentPipelineContext) -> None:
	if ctx.probabilistic_overall_metrics is None or ctx.probabilistic_overall_metrics.empty:
		return
	write_csv(
		ctx.artifacts_paths_map.evaluation_file("probabilistic_overall_model_evaluation.csv"),
		ctx.probabilistic_overall_metrics,
	)


def write_probabilistic_metrics_by_horizon_evaluation(ctx: ExperimentPipelineContext) -> None:
	if ctx.probabilistic_by_horizon_metrics is None or ctx.probabilistic_by_horizon_metrics.empty:
		return
	write_csv(
		ctx.artifacts_paths_map.evaluation_file("probabilistic_metrics_by_horizon.csv"),
		ctx.probabilistic_by_horizon_metrics,
	)


def write_metrics_by_segment_evaluation(ctx: ExperimentPipelineContext) -> None:
	if ctx.evaluation is None:
		return
	write_csv(
		ctx.artifacts_paths_map.evaluation_file("metrics_by_segment.csv"),
		ctx.evaluation.metrics_by_segment,
	)


def write_metrics_by_category_evaluation(ctx: ExperimentPipelineContext) -> None:
	if ctx.evaluation is None:
		return
	write_csv(
		ctx.artifacts_paths_map.evaluation_file("metrics_by_category.csv"),
		ctx.evaluation.metrics_by_category,
	)


def write_probabilistic_metrics_by_segment_evaluation(ctx: ExperimentPipelineContext) -> None:
	if ctx.evaluation is None:
		return
	write_csv(
		ctx.artifacts_paths_map.evaluation_file("probabilistic_metrics_by_segment.csv"),
		ctx.evaluation.probabilistic_metrics_by_segment,
	)


def write_probabilistic_metrics_by_category_evaluation(ctx: ExperimentPipelineContext) -> None:
	if ctx.evaluation is None:
		return
	write_csv(
		ctx.artifacts_paths_map.evaluation_file("probabilistic_metrics_by_category.csv"),
		ctx.evaluation.probabilistic_metrics_by_category,
	)


def write_probabilistic_calibration_summary(ctx: ExperimentPipelineContext) -> None:
	if ctx.evaluation is None:
		return
	write_csv(
		ctx.artifacts_paths_map.evaluation_file("probabilistic_calibration_summary.csv"),
		ctx.evaluation.probabilistic_calibration_summary,
	)


def write_bootstrap_ci_model_differences(ctx: ExperimentPipelineContext) -> None:
	if ctx.evaluation is None:
		return
	write_csv(
		ctx.artifacts_paths_map.evaluation_file("bootstrap_ci_model_differences.csv"),
		ctx.evaluation.bootstrap_ci,
	)


def write_selected_error_cases(ctx: ExperimentPipelineContext) -> None:
	if ctx.evaluation is None:
		return
	write_csv(
		ctx.artifacts_paths_map.evaluation_file("selected_error_cases.csv"),
		ctx.evaluation.error_cases,
	)


def write_rolling_vs_holdout_diagnostic(ctx: ExperimentPipelineContext) -> None:
	if ctx.evaluation is None:
		return
	write_csv(
		ctx.artifacts_paths_map.backtest_file("rolling_vs_holdout_diagnostic.csv"),
		ctx.evaluation.rolling_vs_holdout,
	)


def write_leader_forecast(ctx: ExperimentPipelineContext) -> None:
	if ctx.predictions is None or ctx.overall_metrics is None or ctx.overall_metrics.empty:
		return

	model_name = ctx.overall_metrics.iloc[0]["model_name"]
	leader_forecast = ctx.predictions[ctx.predictions["model_name"] == model_name].copy()
	write_csv(ctx.artifacts_paths_map.evaluation_file("leader_forecast.csv"), leader_forecast)
