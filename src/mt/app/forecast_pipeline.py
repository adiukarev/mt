from mt.app.forecast_stages.raw_dataset_loading import ForecastRawDatasetLoadingPipelineStage
from mt.app.forecast_stages.dataset_preparation import ForecastDatasetPreparationPipelineStage
from mt.app.forecast_stages.frame_building import ForecastFrameBuildingPipelineStage
from mt.app.forecast_stages.artifact_persistence import ForecastArtifactPersistencePipelineStage
from mt.app.forecast_stages.reference_model_resolution import \
	ForecastReferenceModelResolutionPipelineStage
from mt.app.forecast_stages.generate_predictions import ForecastGeneratePredictionsPipelineStage
from mt.domain.pipeline.pipeline import BasePipeline
from mt.domain.forecast.forecast_pipeline_context import ForecastPipelineContext
from mt.domain.forecast.forecast_pipeline_manifest import ForecastPipelineManifest


class ForecastPipeline(BasePipeline):
	def __init__(self) -> None:
		super().__init__([
			ForecastReferenceModelResolutionPipelineStage(),
			ForecastRawDatasetLoadingPipelineStage(),
			ForecastDatasetPreparationPipelineStage(),
			ForecastFrameBuildingPipelineStage(),
			ForecastGeneratePredictionsPipelineStage(),
			ForecastArtifactPersistencePipelineStage(),
		])

	def build_context(self, manifest: ForecastPipelineManifest) -> ForecastPipelineContext:
		return ForecastPipelineContext(manifest=manifest)

	def finalize(self, ctx: ForecastPipelineContext) -> ForecastPipelineContext:
		self._persist_run_artifacts(ctx)

		return ctx
