from mt.domain.audit.audit_pipeline_context import AuditPipelineContext
from mt.infra.audit_artifact.preparation.table import write_preparation_artifacts
from mt.infra.audit_artifact.report.text import write_audit_report
from mt.infra.audit_artifact.run.text import write_audit_run_artifacts
from mt.infra.audit_artifact.series.plot import (
	write_series_acf_pacf_plots,
	write_series_aggregation_granularity_plots,
	write_series_decomposition_plots,
	write_series_feature_overlay_plots,
	write_series_feature_overlay_recent_plots,
	write_series_noise_comparison_plots,
	write_series_outlier_plots,
	write_series_sales_distribution_plots,
	write_series_seasonal_naive_overlay_plots,
	write_series_smoothed_window_10_plots,
)
from mt.infra.audit_artifact.series.table import write_series_artifacts


def write_audit_artifacts(ctx: AuditPipelineContext) -> None:
	if ctx.dataset is None or ctx.segments is None or ctx.audit_artifacts is None:
		raise ValueError()

	audit_artifacts = ctx.audit_artifacts

	# preparation
	write_preparation_artifacts(ctx, audit_artifacts)

	# series
	write_series_artifacts(ctx, audit_artifacts)
	write_series_feature_overlay_plots(ctx)
	write_series_feature_overlay_recent_plots(ctx)
	write_series_seasonal_naive_overlay_plots(ctx)
	write_series_outlier_plots(ctx)
	write_series_smoothed_window_10_plots(ctx)
	write_series_noise_comparison_plots(ctx)
	write_series_aggregation_granularity_plots(ctx)
	write_series_sales_distribution_plots(ctx)
	write_series_acf_pacf_plots(ctx)
	write_series_decomposition_plots(ctx)

	# report
	write_audit_report(ctx, audit_artifacts)

	# run
	write_audit_run_artifacts(ctx, audit_artifacts)
