from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mt.domain.backtest.backtest_manifest import BacktestManifest
from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.dataset.dataset_source_manifest import DatasetSourceManifest
from mt.domain.feature.feature_manifest import FeatureManifest
from mt.domain.feature.feature_set import FeatureSet, max_feature_set
from mt.domain.model.model_manifest import ModelManifest
from mt.domain.model.model_name import ModelName, normalize_model_name
from mt.domain.runtime.runtime_manifest import RuntimeManifest
from mt.infra.artifact.text_reader import read_yaml_mapping
from mt.infra.helper.dict import get_required_mapping_section


@dataclass(slots=True)
class ExperimentPipelineManifest:
	dataset: DatasetManifest = field(default_factory=DatasetManifest)
	source: DatasetSourceManifest | None = None
	backtest: BacktestManifest = field(default_factory=BacktestManifest)
	runtime: RuntimeManifest = field(default_factory=RuntimeManifest)
	models: list[ModelManifest] = field(default_factory=list)

	def __post_init__(self) -> None:
		if not self.models:
			raise ValueError()
		if self.backtest.horizon_start < 1 or self.backtest.horizon_end > 8:
			raise ValueError()

		names = [model.name for model in self.models]
		if len(set(names)) != len(names):
			raise ValueError()

	@property
	def enabled_models(self) -> list[ModelManifest]:
		return [model for model in self.models if model.enabled]

	@property
	def enabled_model_names(self) -> list[ModelName]:
		return [model.name for model in self.enabled_models]

	def get_model(self, model_name: str | ModelName) -> ModelManifest:
		normalized_name = normalize_model_name(model_name)
		for model in self.models:
			if model.name == normalized_name:
				return model
		raise KeyError(normalized_name)

	def get_enabled_model(self, model_name: str | ModelName) -> ModelManifest:
		model = self.get_model(model_name)
		if not model.enabled:
			raise ValueError(f"Model '{model_name}' is disabled")
		return model

	def build_combined_feature_manifest(self) -> FeatureManifest:
		enabled_feature_manifests = [
			model.features
			for model in self.enabled_models
			if model.features.enabled
		]
		if not enabled_feature_manifests:
			return FeatureManifest(enabled=False, feature_set=FeatureSet.F0)

		return FeatureManifest(
			enabled=True,
			feature_set=max_feature_set([manifest.feature_set for manifest in enabled_feature_manifests]),
			lags=sorted({lag for manifest in enabled_feature_manifests for lag in manifest.lags}),
			rolling_windows=sorted(
				{window for manifest in enabled_feature_manifests for window in manifest.rolling_windows}
			),
			use_calendar=any(manifest.use_calendar for manifest in enabled_feature_manifests),
			use_category_encodings=any(
				manifest.use_category_encodings for manifest in enabled_feature_manifests
			),
		)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)

	@staticmethod
	def from_dict(data: dict[str, Any]) -> "ExperimentPipelineManifest":
		source_data = data.get("source")
		return ExperimentPipelineManifest(
			dataset=DatasetManifest(**get_required_mapping_section(data, "dataset")),
			source=DatasetSourceManifest(**dict(source_data)) if isinstance(source_data, dict) else None,
			backtest=BacktestManifest(**get_required_mapping_section(data, "backtest")),
			runtime=RuntimeManifest(**get_required_mapping_section(data, "runtime")),
			models=[ModelManifest.load(item) for item in data.get("models")],
		)

	@staticmethod
	def load(source: str | Path | dict[str, Any]) -> "ExperimentPipelineManifest":
		if isinstance(source, dict):
			return ExperimentPipelineManifest.from_dict(source)

		return ExperimentPipelineManifest.from_dict(read_yaml_mapping(source))
