from dataclasses import dataclass, field

from pandas import DataFrame

from mt.domain.dataset.dataset import DatasetBundle, DatasetLoadData
from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.domain.forecast.forecast_artifact import ForecastArtifactPathsMap
from mt.domain.forecast.forecast_pipeline_manifest import ForecastPipelineManifest
from mt.infra.forecast.reference_model import ReferenceModelConfig


@dataclass(slots=True)
class ForecastPipelineContext(BasePipelineContext):
	manifest: ForecastPipelineManifest
	artifacts_paths_map: ForecastArtifactPathsMap | None = None

	resolved_dataset_manifest: DatasetManifest | None = None
	raw_dataset: DatasetLoadData | None = None
	dataset: DatasetBundle | None = None

	frame: DataFrame | None = None
	reference_model: ReferenceModelConfig | None = None
	resolved_horizon: int | None = None
	predictions: DataFrame | None = None

	metrics: DataFrame | None = None
	probabilistic_metrics: DataFrame | None = None

	def __post_init__(self) -> None:
		self.artifacts_paths_map = ForecastArtifactPathsMap.ensure(
			self.manifest.runtime.artifacts_dir)

	def require_resolved_dataset_manifest(self) -> DatasetManifest:
		return self.require_value(self.resolved_dataset_manifest, "resolved_dataset_manifest")

	def require_raw_dataset(self) -> DatasetLoadData:
		return self.require_value(self.raw_dataset, "raw_dataset")

	def require_dataset(self) -> DatasetBundle:
		return self.require_value(self.dataset, "dataset")

	def require_frame(self) -> DataFrame:
		return self.require_value(self.frame, "frame")

	def require_reference_model(self) -> ReferenceModelConfig:
		return self.require_value(self.reference_model, "reference_model")

	def require_predictions(self) -> DataFrame:
		return self.require_value(self.predictions, "predictions")

	def require_metrics(self) -> DataFrame:
		return self.require_value(self.metrics, "metrics")
