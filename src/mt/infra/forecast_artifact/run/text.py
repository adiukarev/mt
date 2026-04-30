from mt.domain.forecast.forecast_pipeline_context import ForecastPipelineContext
from mt.domain.probabilistic.probabilistic import ProbabilisticColumn, ProbabilisticStatus
from mt.infra.artifact.text_writer import write_yaml
from mt.infra.observability.runtime.summary_builder import build_tracking_summary


def write_forecast_run_artifacts(ctx: ForecastPipelineContext) -> None:
	reference_model = ctx.require_reference_model()
	predictions = ctx.require_predictions()
	dataset = ctx.require_dataset()

	payload = {
		"dataset_kind": dataset.kind,
		"reference_model_name": reference_model.model_name,
		"resolved_horizon": ctx.resolved_horizon,
		"prediction_rows": len(predictions),
		"predicted_series_count": int(predictions["series_id"].nunique()),
		"model_source_kind": ctx.manifest.model.resolve_source_kind(
			ctx.observability.execution_mode if ctx.observability else None
		),
		"model_source_dir": ctx.manifest.model.local.model_dir,
		"registry_selection_alias": ctx.manifest.model.registry.selection.alias,
		"registry_selection_dag_id": ctx.manifest.model.registry.selection.dag_id,
		"resolved_model_source": reference_model.source_descriptor,
		"inference_mode": ctx.runtime_metadata.get("forecast_inference_mode"),
		"inference_reasons": ctx.runtime_metadata.get("forecast_inference_reasons"),
		"probabilistic_summary": {
			"available_rows": int(
				(predictions[ProbabilisticColumn.STATUS].astype(str) == ProbabilisticStatus.AVAILABLE).sum()
			)
			if ProbabilisticColumn.STATUS in predictions.columns
			else 0,
			"available_share": float(
				(predictions[ProbabilisticColumn.STATUS].astype(str) == ProbabilisticStatus.AVAILABLE).mean()
			)
			if ProbabilisticColumn.STATUS in predictions.columns
			else 0.0,
			"metric_rows": int(len(ctx.probabilistic_metrics))
			if ctx.probabilistic_metrics is not None
			else 0,
		},
		**build_tracking_summary(ctx),
	}
	write_yaml(ctx.artifacts_paths_map.run_file("summary.yaml"), payload)
