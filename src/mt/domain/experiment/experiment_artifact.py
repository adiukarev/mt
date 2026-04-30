from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mt.infra.model_artifact.model.fitting import TrainedModelBundle
from mt.domain.model.model_artifact import ModelArtifactData
from mt.domain.model.model_manifest import ModelManifest
from mt.domain.model.model_result import ModelResult

from mt.domain.artifact.artifact import BaseArtifactPathsMap


@dataclass(slots=True)
class ExperimentModelArtifactPayload:
	model_dir: Path
	result: ModelResult
	model_manifest: ModelManifest
	model_artifact: ModelArtifactData | None
	metrics_overall: pd.DataFrame
	metrics_by_horizon: pd.DataFrame
	probabilistic_metrics_overall: pd.DataFrame
	probabilistic_metrics_by_horizon: pd.DataFrame


@dataclass(slots=True)
class ExperimentArtifactPathsMap(BaseArtifactPathsMap):
	evaluation: Path
	preparation: Path
	model: Path
	plots: Path
	backtest: Path

	@classmethod
	def ensure(cls, root: str | Path) -> "ExperimentArtifactPathsMap":
		root = Path(root)

		artifact_paths_map = cls(
			**cls._paths(root),
			evaluation=root / "evaluation",
			preparation=root / "preparation",
			model=root / "model",
			plots=root / "plots",
			backtest=root / "backtest",
		)

		cls._ensure_dirs(
			artifact_paths_map.root,
			artifact_paths_map.report,
			artifact_paths_map.run,
			artifact_paths_map.evaluation,
			artifact_paths_map.preparation,
			artifact_paths_map.model,
			artifact_paths_map.plots,
			artifact_paths_map.backtest,
		)

		return artifact_paths_map

	def preparation_file(self, filename: str) -> Path:
		return self.preparation / filename

	def model_file(self, filename: str) -> Path:
		return self.model / filename

	def backtest_file(self, filename: str) -> Path:
		return self.backtest / filename

	def evaluation_file(self, filename: str) -> Path:
		return self.evaluation / filename

	def plot_file(self, filename: str) -> Path:
		return self.plots / filename
