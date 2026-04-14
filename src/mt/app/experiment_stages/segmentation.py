from mt.app.base_stages.segmentation import BaseSegmentationStage
from mt.domain.experiment import ExperimentPipelineContext
from mt.infra.artifact.writer import write_markdown
from mt.infra.artifact.experiment_paths import experiment_artifact_relpath


class SegmentationStage(BaseSegmentationStage):
	name = "experiment_segmentation"

	def after_execute(self, ctx: ExperimentPipelineContext) -> None:
		if ctx.segments is None:
			raise ValueError()

		write_markdown(
			ctx.artifacts_paths_map.root / experiment_artifact_relpath(f"{self.name}.md"),
			[
				"# Сегментация рядов",
				"",
				f"- число рядов: {len(ctx.segments)}",
				f"- число сегментов: {ctx.segments['segment_label'].nunique()}",
			]
		)
