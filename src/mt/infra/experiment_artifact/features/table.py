import pandas as pd

from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.infra.artifact.text_writer import write_csv


def write_feature_registry(ctx: ExperimentPipelineContext) -> None:
	write_csv(
		ctx.artifacts_paths_map.preparation_file("06_feature_registry.csv"),
		ctx.require_feature_registry(),
	)


def write_feature_block_summary(ctx: ExperimentPipelineContext) -> None:
	registry = ctx.require_feature_registry().copy()
	summary = (
		registry.groupby("group", as_index=False)
		.agg(
			feature_count=("name", "count"),
			enabled_count=("enabled", lambda values: int(pd.Series(values).fillna(False).astype(bool).sum())),
			disabled_count=("enabled", lambda values: int((~pd.Series(values).fillna(False).astype(bool)).sum())),
			forecast_available_count=(
				"availability_at_forecast_time",
				lambda values: int(pd.Series(values).fillna(False).astype(bool).sum()),
			),
			known_in_advance_count=(
				"covariate_class",
				lambda values: int((pd.Series(values) == "known_in_advance").sum()),
			),
			observed_count=(
				"covariate_class",
				lambda values: int((pd.Series(values) == "observed").sum()),
			),
		)
	)
	summary.insert(0, "feature_set", str(ctx.feature_manifest.feature_set))
	summary.insert(1, "supervised_feature_column_count", len(ctx.feature_columns))
	write_csv(
		ctx.artifacts_paths_map.preparation_file("07_feature_block_summary.csv"),
		summary,
	)


def write_model_feature_usage(ctx: ExperimentPipelineContext) -> None:
	if not ctx.model_feature_usage_rows:
		return
	write_csv(
		ctx.artifacts_paths_map.preparation_file("08_model_feature_usage.csv"),
		pd.DataFrame(ctx.model_feature_usage_rows),
	)
