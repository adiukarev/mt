from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.infra.artifact.text_writer import write_yaml
from mt.infra.observability.runtime.summary_builder import build_tracking_summary


def write_summary(ctx: ExperimentPipelineContext) -> None:
	dataset = ctx.require_dataset()
	selected_model_name = ctx.require_selected_model_name()
	selected_model_metrics = ctx.require_selected_model_metrics()
	evaluation = ctx.require_evaluation()
	probabilistic_overall_metrics = ctx.require_probabilistic_overall_metrics()
	probabilistic_by_horizon_metrics = ctx.require_probabilistic_by_horizon_metrics()

	payload = {
		"dataset_kind": dataset.kind,
		"aggregation_level": dataset.aggregation_level,
		"target_name": dataset.target_name,
		"selected_model_name": selected_model_name,
		"final_model_dir": str(ctx.artifacts_paths_map.model),
		"evaluation_table": str(
			ctx.artifacts_paths_map.evaluation_file("overall_model_evaluation.csv")
		),
		"metrics": selected_model_metrics,
		"probabilistic_evaluation_table": str(
			ctx.artifacts_paths_map.evaluation_file("probabilistic_overall_model_evaluation.csv")
		),
		"probabilistic_summary": {
			"available_model_count": int(len(probabilistic_overall_metrics)),
			"by_horizon_rows": int(len(probabilistic_by_horizon_metrics)),
			"calibration_rows": int(len(evaluation.probabilistic_calibration_summary)),
		},
		"backtest": {
			"horizon_start": ctx.manifest.backtest.horizon_start,
			"horizon_end": ctx.manifest.backtest.horizon_end,
			"shared_origin_grid": ctx.manifest.backtest.shared_origin_grid,
		},
		"source_descriptor": ctx.source_descriptor,
		**build_tracking_summary(ctx),
	}
	write_yaml(ctx.artifacts_paths_map.run_file("summary.yaml"), payload)
