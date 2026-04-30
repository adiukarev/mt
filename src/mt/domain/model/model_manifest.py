from dataclasses import dataclass, field
from typing import Any

from mt.domain.model.model_config_manifest import (
	ModelConfigManifest,
	build_model_config,
	MODEL_CONFIG_TYPES,
)
from mt.domain.model.model_family import MODEL_FAMILY_BY_NAME, ModelFamily
from mt.domain.model.model_name import ALLOWED_MODEL_NAMES, ModelName, normalize_model_name
from mt.domain.feature.feature_manifest import FeatureManifest


@dataclass(slots=True)
class ModelManifest:
	name: ModelName
	enabled: bool = True
	features: FeatureManifest = field(default_factory=FeatureManifest)
	config: ModelConfigManifest | None = None

	def __post_init__(self) -> None:
		self.name = normalize_model_name(self.name)
		if self.name not in ALLOWED_MODEL_NAMES:
			raise ValueError()
		if self.config is not None and not isinstance(self.config, MODEL_CONFIG_TYPES):
			raise ValueError()

	@property
	def family(self) -> ModelFamily:
		return MODEL_FAMILY_BY_NAME.get(self.name, ModelFamily.OTHER)

	@staticmethod
	def load(item: object) -> "ModelManifest":
		if not isinstance(item, dict):
			raise TypeError()

		data = dict(item)
		name = normalize_model_name(data.pop("name"))
		features_data = data.pop("features", {})
		config_data = data.pop("config", None)
		if config_data is not None and not isinstance(config_data, dict):
			raise TypeError()

		return ModelManifest(
			**data,
			name=name,
			features=FeatureManifest(**features_data),
			config=build_model_config(name, config_data),
		)
