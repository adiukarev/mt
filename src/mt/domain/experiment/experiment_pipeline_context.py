from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mt.domain.dataset.dataset import DatasetBundle, DatasetLoadData
from mt.domain.experiment.experiment_artifact import ExperimentArtifactPathsMap
from mt.domain.experiment.experiment_artifact import ExperimentModelArtifactPayload
from mt.domain.experiment.experiment_pipeline_manifest import ExperimentPipelineManifest
from mt.domain.feature.feature_manifest import FeatureManifest
from mt.domain.model.model_manifest import ModelManifest
from mt.domain.model.model_name import ModelName
from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.infra.analysis.comparison import EvaluationArtifacts
from mt.infra.model_artifact.model.fitting import TrainedModelBundle


@dataclass(slots=True)
class ExperimentPipelineContext(BasePipelineContext):
	manifest: ExperimentPipelineManifest
	artifacts_paths_map: ExperimentArtifactPathsMap | None = None

	raw_dataset: DatasetLoadData | None = None
	dataset: DatasetBundle | None = None
	source_descriptor: dict[str, object] = field(default_factory=dict)

	feature_registry: pd.DataFrame | None = None
	feature_columns: list[str] = field(default_factory=list)
	feature_manifest: FeatureManifest = field(default_factory=FeatureManifest)

	segments: pd.DataFrame | None = None
	supervised: pd.DataFrame | None = None
	windows: pd.DataFrame | None = None
	predictions: pd.DataFrame | None = None

	overall_metrics: pd.DataFrame | None = None
	by_horizon_metrics: pd.DataFrame | None = None
	probabilistic_overall_metrics: pd.DataFrame | None = None
	probabilistic_by_horizon_metrics: pd.DataFrame | None = None

	evaluation: EvaluationArtifacts | None = None

	selected_model_name: ModelName | None = None
	selected_model_metrics: dict[str, Any] = field(default_factory=dict)
	best_model_bundle: TrainedModelBundle | None = None

	run_catalog_rows: list[dict[str, object]] = field(default_factory=list)

	model_catalog_rows: list[dict[str, object]] = field(default_factory=list)
	model_feature_usage_rows: list[dict[str, object]] = field(default_factory=list)
	model_manifests: list[ModelManifest] = field(default_factory=list)
	model_artifact_payloads: list[ExperimentModelArtifactPayload] = field(default_factory=list)

	def __post_init__(self) -> None:
		self.artifacts_paths_map = ExperimentArtifactPathsMap.ensure(
			self.manifest.runtime.artifacts_dir)

	@property
	def dataset_manifest(self) -> object:
		return self.manifest.dataset

	@property
	def backtest_manifest(self) -> object:
		return self.manifest.backtest

	def require_dataset(self) -> DatasetBundle:
		return self.require_value(self.dataset, "dataset")

	def require_raw_dataset(self) -> DatasetLoadData:
		return self.require_value(self.raw_dataset, "raw_dataset")

	def require_feature_registry(self) -> pd.DataFrame:
		return self.require_value(self.feature_registry, "feature_registry")

	def require_segments(self) -> pd.DataFrame:
		return self.require_value(self.segments, "segments")

	def require_supervised(self) -> pd.DataFrame:
		return self.require_value(self.supervised, "supervised")

	def require_windows(self) -> pd.DataFrame:
		return self.require_value(self.windows, "windows")

	def require_predictions(self) -> pd.DataFrame:
		return self.require_value(self.predictions, "predictions")

	def require_overall_metrics(self) -> pd.DataFrame:
		return self.require_value(self.overall_metrics, "overall_metrics")

	def require_by_horizon_metrics(self) -> pd.DataFrame:
		return self.require_value(self.by_horizon_metrics, "by_horizon_metrics")

	def require_probabilistic_overall_metrics(self) -> pd.DataFrame:
		return self.require_value(
			self.probabilistic_overall_metrics,
			"probabilistic_overall_metrics",
		)

	def require_probabilistic_by_horizon_metrics(self) -> pd.DataFrame:
		return self.require_value(
			self.probabilistic_by_horizon_metrics,
			"probabilistic_by_horizon_metrics",
		)

	def require_evaluation(self) -> EvaluationArtifacts:
		return self.require_value(self.evaluation, "evaluation")

	def require_selected_model_name(self) -> ModelName:
		return self.require_value(self.selected_model_name, "selected_model_name")

	def require_selected_model_metrics(self) -> dict[str, Any]:
		if not self.selected_model_metrics:
			raise ValueError("selected_model_metrics must be available")
		return self.selected_model_metrics

	def require_selected_model_artifact_payload(self) -> ExperimentModelArtifactPayload:
		selected_model_name = self.require_selected_model_name()
		for payload in self.model_artifact_payloads:
			if str(payload.result.info.model_name) == str(selected_model_name):
				return payload
		raise ValueError(
			f"Selected experiment model payload is not available for {selected_model_name}"
		)

	def require_best_model_bundle(self) -> TrainedModelBundle:
		return self.require_value(self.best_model_bundle, "best_model_bundle")
