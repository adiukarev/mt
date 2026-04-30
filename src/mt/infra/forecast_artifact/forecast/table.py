from mt.domain.forecast.forecast_pipeline_context import ForecastPipelineContext
from mt.domain.probabilistic.probabilistic import ProbabilisticColumn, QUANTILE_COLUMNS
from mt.infra.artifact.text_writer import write_csv


def write_predictions(ctx: ForecastPipelineContext) -> None:
	if ctx.predictions is None:
		raise ValueError()

	write_csv(ctx.artifacts_paths_map.forecast_file("predictions.csv"), ctx.predictions)
	write_csv(
		ctx.artifacts_paths_map.forecast_file("quantiles.csv"),
		ctx.predictions.loc[
			:,
			[
				"model_name",
				"model_family",
				"series_id",
				"category",
				"segment_label",
				"forecast_origin",
				"target_date",
				"horizon",
				"actual",
				"prediction",
				*QUANTILE_COLUMNS,
				ProbabilisticColumn.SOURCE,
				ProbabilisticColumn.STATUS,
			],
		],
	)
	write_csv(
		ctx.artifacts_paths_map.forecast_file("intervals.csv"),
		ctx.predictions.loc[
			:,
			[
				"model_name",
				"model_family",
				"series_id",
				"category",
				"segment_label",
				"forecast_origin",
				"target_date",
				"horizon",
				"actual",
				"prediction",
				ProbabilisticColumn.LO_80,
				ProbabilisticColumn.HI_80,
				ProbabilisticColumn.LO_95,
				ProbabilisticColumn.HI_95,
				ProbabilisticColumn.SOURCE,
				ProbabilisticColumn.STATUS,
			],
		],
	)


def write_metrics_by_horizon(ctx: ForecastPipelineContext) -> None:
	if ctx.metrics is None:
		raise ValueError()

	write_csv(ctx.artifacts_paths_map.forecast_file("metrics_by_horizon.csv"), ctx.metrics)
	if ctx.probabilistic_metrics is not None:
		write_csv(
			ctx.artifacts_paths_map.forecast_file("probabilistic_metrics_by_horizon.csv"),
			ctx.probabilistic_metrics,
		)
