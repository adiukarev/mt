from mt.domain.forecast.forecast_pipeline_context import ForecastPipelineContext
from mt.infra.artifact.text_writer import write_csv


def write_filtered_dataset(ctx: ForecastPipelineContext) -> None:
	if ctx.frame is None:
		raise ValueError()

	write_csv(ctx.artifacts_paths_map.dataset_file("filtered_dataset.csv"), ctx.frame)
