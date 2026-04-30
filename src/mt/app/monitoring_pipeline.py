from mt.app.monitoring_stages.dataset_refresh import MonitoringDatasetRefreshPipelineStage
from mt.app.monitoring_stages.decision_artifact import MonitoringDecisionArtifactPipelineStage
from mt.app.monitoring_stages.drift_detection import MonitoringDriftDetectionPipelineStage
from mt.app.monitoring_stages.inference_on_recent_actuals import (
	MonitoringInferenceOnRecentActualsPipelineStage,
)
from mt.app.monitoring_stages.artifact_persistence import MonitoringArtifactPersistencePipelineStage
from mt.app.monitoring_stages.quality_gate import MonitoringQualityGatePipelineStage
from mt.app.monitoring_stages.reference_model_resolution import (
	MonitoringReferenceModelResolutionPipelineStage,
)
from mt.app.monitoring_stages.source_resolution import MonitoringSourceResolutionPipelineStage
from mt.domain.monitoring.monitoring_pipeline_context import MonitoringPipelineContext
from mt.domain.monitoring.monitoring_pipeline_manifest import MonitoringPipelineManifest
from mt.domain.pipeline.pipeline import BasePipeline


class MonitoringPipeline(BasePipeline):
	def __init__(self) -> None:
		super().__init__([
			MonitoringSourceResolutionPipelineStage(),
			MonitoringDatasetRefreshPipelineStage(),
			MonitoringReferenceModelResolutionPipelineStage(),
			MonitoringInferenceOnRecentActualsPipelineStage(),
			MonitoringDriftDetectionPipelineStage(),
			MonitoringQualityGatePipelineStage(),
			MonitoringDecisionArtifactPipelineStage(),
			MonitoringArtifactPersistencePipelineStage(),
		])

	def build_context(self, manifest: MonitoringPipelineManifest) -> MonitoringPipelineContext:
		return MonitoringPipelineContext(manifest=manifest)

	def finalize(self, ctx: MonitoringPipelineContext) -> MonitoringPipelineContext:
		self._persist_run_artifacts(ctx)
		return ctx
