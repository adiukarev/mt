from mt.domain.monitoring.monitoring_pipeline_context import MonitoringPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.dataset.sources.source_resolver import infer_dataset_source_manifest


class MonitoringSourceResolutionPipelineStage(BasePipelineStage):
	def execute(self, ctx: MonitoringPipelineContext) -> None:
		source_manifest = infer_dataset_source_manifest(ctx.manifest.dataset)
		ctx.source_descriptor = {
			"source_type": source_manifest.source_type.value,
			"dataset_path": ctx.manifest.dataset.path,
			"source_inference": "dataset_kind",
		}
