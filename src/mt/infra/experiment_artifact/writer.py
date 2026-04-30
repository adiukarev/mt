from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.infra.experiment_artifact.comparison.table import (
	write_bootstrap_ci_model_differences,
	write_leader_forecast,
	write_metrics_by_category_evaluation,
	write_metrics_by_horizon_evaluation,
	write_metrics_by_segment_evaluation,
	write_overall_model_evaluation,
	write_probabilistic_calibration_summary,
	write_probabilistic_metrics_by_category_evaluation,
	write_probabilistic_metrics_by_horizon_evaluation,
	write_probabilistic_metrics_by_segment_evaluation,
	write_probabilistic_overall_model_evaluation,
	write_rolling_vs_holdout_diagnostic,
	write_selected_error_cases,
)
from mt.infra.experiment_artifact.preparation.table import (
	write_preparation_artifacts,
)
from mt.infra.experiment_artifact.features.table import (
	write_feature_block_summary,
	write_feature_registry,
	write_model_feature_usage,
)
from mt.infra.experiment_artifact.models.model_artifact_writer import write_experiment_model_artifacts
from mt.infra.experiment_artifact.plots.plot import (
	write_bias_by_horizon_plot,
	write_calibration_curve_plot,
	write_coverage_by_horizon_plot,
	write_error_distribution_model_plot,
	write_fan_chart_leader,
	write_interval_width_by_horizon_plot,
	write_model_ranking_wape_plot,
	write_segment_model_comparison_plot,
	write_smape_by_horizon_plot,
	write_wape_by_horizon_plot,
	write_wape_heatmap_model_horizon_plot,
	write_wis_by_horizon_plot,
)
from mt.infra.experiment_artifact.report.text import write_comparison_report
from mt.infra.experiment_artifact.run.text import write_summary
from mt.infra.experiment_artifact.validation.plot import write_rolling_backtest_schematic
from mt.infra.experiment_artifact.validation.table import (
	write_backtest_window_generation_summary,
	write_backtest_window_summary,
	write_backtest_window_train_test_counts,
	write_backtest_windows,
	write_backtest_windows_by_horizon,
)


def write_experiment_artifacts(ctx: ExperimentPipelineContext) -> None:
	# evaluation
	write_overall_model_evaluation(ctx)
	write_metrics_by_horizon_evaluation(ctx)
	write_metrics_by_segment_evaluation(ctx)
	write_metrics_by_category_evaluation(ctx)
	write_probabilistic_overall_model_evaluation(ctx)
	write_probabilistic_metrics_by_horizon_evaluation(ctx)
	write_probabilistic_metrics_by_segment_evaluation(ctx)
	write_probabilistic_metrics_by_category_evaluation(ctx)
	write_probabilistic_calibration_summary(ctx)
	write_bootstrap_ci_model_differences(ctx)
	write_selected_error_cases(ctx)
	write_leader_forecast(ctx)

	# preparation
	write_preparation_artifacts(ctx)
	write_feature_registry(ctx)
	write_feature_block_summary(ctx)
	write_model_feature_usage(ctx)

	# models
	write_experiment_model_artifacts(ctx)

	# plots
	write_model_ranking_wape_plot(ctx)
	write_wape_by_horizon_plot(ctx)
	write_smape_by_horizon_plot(ctx)
	write_bias_by_horizon_plot(ctx)
	write_wape_heatmap_model_horizon_plot(ctx)
	write_segment_model_comparison_plot(ctx)
	write_error_distribution_model_plot(ctx)
	write_coverage_by_horizon_plot(ctx)
	write_interval_width_by_horizon_plot(ctx)
	write_wis_by_horizon_plot(ctx)
	write_calibration_curve_plot(ctx)
	write_fan_chart_leader(ctx)

	# report
	write_comparison_report(ctx)

	# run
	write_summary(ctx)

	# validation
	write_backtest_windows(ctx)
	write_backtest_windows_by_horizon(ctx)
	write_backtest_window_summary(ctx)
	write_backtest_window_train_test_counts(ctx)
	write_backtest_window_generation_summary(ctx)
	write_rolling_vs_holdout_diagnostic(ctx)
	write_rolling_backtest_schematic(ctx)
