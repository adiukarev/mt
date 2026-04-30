from mt.domain.forecast.forecast_pipeline_context import ForecastPipelineContext
from mt.infra.forecast_artifact.dataset.table import write_filtered_dataset
from mt.infra.forecast_artifact.forecast.table import write_metrics_by_horizon, write_predictions
from mt.infra.forecast_artifact.plots.plot import (
	write_actual_vs_prediction_plot,
	write_overlay_plots,
)
from mt.infra.forecast_artifact.report.text import write_forecast_report
from mt.infra.forecast_artifact.run.text import write_forecast_run_artifacts


def write_forecast_artifacts(ctx: ForecastPipelineContext) -> None:
	# dataset
	write_filtered_dataset(ctx)

	# forecast
	write_predictions(ctx)
	write_metrics_by_horizon(ctx)

	# plots
	write_overlay_plots(ctx)
	write_actual_vs_prediction_plot(ctx.predictions, ctx.artifacts_paths_map.plots)

	# report
	write_forecast_report(ctx)

	# run
	write_forecast_run_artifacts(ctx)
