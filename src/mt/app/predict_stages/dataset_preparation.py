from __future__ import annotations

from time import perf_counter

from mt.domain.predict import PredictPipelineContext
from mt.domain.stage import BaseStage
from mt.infra.artifact.writer import write_csv, write_markdown
from mt.infra.synthetic.predict import prepare_predict_frame


class PredictDatasetPreparationStage(BaseStage):
	name = "predict_dataset_preparation"

	def execute(self, ctx: PredictPipelineContext) -> None:
		if ctx.reference_model is None:
			raise ValueError()

		ctx.frame = prepare_predict_frame(
			dataset_path=ctx.manifest.input.dataset_path,
			scenario_name=ctx.manifest.input.scenario_name,
			aggregation_level=ctx.reference_model.artifact.training_aggregation_level
			if ctx.reference_model.artifact is not None
			else None,
		)

		# artifacts
		self._persist_artifacts(ctx)

	def _persist_artifacts(self, ctx: PredictPipelineContext) -> None:
		write_csv(ctx.artifacts_paths_map.dataset / "filtered_dataset.csv", ctx.frame)
		write_markdown(
			ctx.artifacts_paths_map.dataset / "dataset_summary.md",
			[
				"# Predict Dataset",
				"",
				f"- rows: {len(ctx.frame)}",
				f"- series_count: {ctx.frame['series_id'].nunique()}",
				f"- aggregation_level: {ctx.reference_model.artifact.training_aggregation_level if ctx.reference_model and ctx.reference_model.artifact is not None else 'raw'}",
				f"- scenario_name: {ctx.manifest.input.scenario_name or 'all'}",
			],
		)
