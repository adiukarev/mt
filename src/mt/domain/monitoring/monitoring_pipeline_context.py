from dataclasses import dataclass, field

import pandas as pd

from mt.domain.monitoring.monitoring_artifact import MonitoringArtifactPathsMap
from mt.domain.monitoring.monitoring_decision_artifact import MonitoringDecisionArtifact
from mt.domain.monitoring.monitoring_decision import MonitoringDecision
from mt.domain.monitoring.monitoring_pipeline_manifest import MonitoringPipelineManifest
from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.infra.forecast.reference_model import ReferenceModelConfig


@dataclass(slots=True)
class MonitoringPipelineContext(BasePipelineContext):
	manifest: MonitoringPipelineManifest
	artifacts_paths_map: MonitoringArtifactPathsMap | None = None

	source_descriptor: dict[str, object] = field(default_factory=dict)
	reference_frame: pd.DataFrame | None = None
	recent_actuals: pd.DataFrame | None = None
	dataset: pd.DataFrame | None = None
	predictions: pd.DataFrame | None = None
	champion_model: ReferenceModelConfig | None = None
	monitoring_issues: list[str] = field(default_factory=list)

	monitoring_metrics: dict[str, float] = field(default_factory=dict)
	quality_gate_summary: dict[str, object] = field(default_factory=dict)
	decision: MonitoringDecision | None = None
	decision_artifact: MonitoringDecisionArtifact | None = None

	def __post_init__(self) -> None:
		self.artifacts_paths_map = MonitoringArtifactPathsMap.ensure(
			self.manifest.runtime.artifacts_dir
		)

	def require_reference_frame(self) -> pd.DataFrame:
		return self.require_value(self.reference_frame, "reference_frame")

	def require_recent_actuals(self) -> pd.DataFrame:
		return self.require_value(self.recent_actuals, "recent_actuals")

	def require_dataset(self) -> pd.DataFrame:
		return self.require_value(self.dataset, "dataset")

	def require_predictions(self) -> pd.DataFrame:
		return self.require_value(self.predictions, "predictions")

	def require_champion_model(self) -> ReferenceModelConfig:
		return self.require_value(self.champion_model, "champion_model")

	def require_decision(self) -> MonitoringDecision:
		return self.require_value(self.decision, "decision")

	def require_decision_artifact(self) -> MonitoringDecisionArtifact:
		return self.require_value(self.decision_artifact, "decision_artifact")
