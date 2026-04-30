from mt.domain.monitoring.monitoring_pipeline_context import MonitoringPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.monitoring.writer import write_monitoring_artifacts


class MonitoringArtifactPersistencePipelineStage(BasePipelineStage):
	def execute(self, ctx: MonitoringPipelineContext) -> None:
		write_monitoring_artifacts(ctx)
