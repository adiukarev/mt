from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.domain.tracking.tracking_contract import FINAL_METRICS_CONTRACT
from mt.infra.tracking.metrics_builders.shared import (
	add_dataset_metrics,
	add_frame_rows_metric,
	add_mapping_metrics,
	add_metrics_table,
	add_prediction_metrics,
	add_stage_metrics,
	resolve_selected_model_name,
	resolve_value_path,
)
from mt.infra.tracking.payload_adapter import add_scalar_metric


def build_final_metrics(ctx: BasePipelineContext) -> dict[str, float]:
	metrics: dict[str, float] = {}
	add_scalar_metric(metrics, "pipeline.executed_stage_count", len(ctx.executed_stages))

	add_stage_metrics(metrics, ctx.stage_timings)
	add_dataset_metrics(metrics, getattr(ctx, "dataset", None))
	add_prediction_metrics(metrics, getattr(ctx, "predictions", None))
	_add_declared_metrics(metrics, ctx)

	return metrics


def _add_declared_metrics(metrics: dict[str, float], ctx: BasePipelineContext) -> None:
	selected_model_name = resolve_selected_model_name(ctx)
	for spec in FINAL_METRICS_CONTRACT.metrics_table_specs:
		add_metrics_table(
			metrics,
			spec.prefix,
			resolve_value_path(ctx, spec.source_path),
			selected_model_name=selected_model_name if spec.use_selected_model_name else None,
		)

	for spec in FINAL_METRICS_CONTRACT.row_count_specs:
		add_frame_rows_metric(metrics, spec.metric_key, resolve_value_path(ctx, spec.source_path))

	for spec in FINAL_METRICS_CONTRACT.mapping_metric_specs:
		if not _all_required_paths_present(ctx, spec.required_paths):
			continue
		add_mapping_metrics(
			metrics,
			spec.prefix,
			resolve_value_path(ctx, spec.source_path),
			skip_keys={"model_name"},
		)


def _all_required_paths_present(ctx: BasePipelineContext, paths: tuple[str, ...]) -> bool:
	for path in paths:
		if resolve_value_path(ctx, path) is None:
			return False
	return True
