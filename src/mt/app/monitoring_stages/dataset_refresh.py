from mt.domain.monitoring.monitoring_pipeline_context import MonitoringPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.dataset.sources.source_resolver import build_dataset_source_service_for_dataset


class MonitoringDatasetRefreshPipelineStage(BasePipelineStage):
	def execute(self, ctx: MonitoringPipelineContext) -> None:
		service = build_dataset_source_service_for_dataset(ctx.manifest.dataset)
		result = service.refresh_dataset(ctx.manifest.dataset)
		ctx.reference_frame = result.reference_frame
		ctx.recent_actuals = result.recent_actuals
		ctx.dataset = result.full_frame
		ctx.source_descriptor.update(result.source_descriptor)
		ctx.source_descriptor["materialized_paths"] = result.materialized_paths
