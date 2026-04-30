from typing import Any, TypeAlias

from mt.domain.model.model_config_dl_manifest import (
	ModelConfigMlpManifest,
	ModelConfigNBeatsManifest,
)
from mt.domain.model.model_config_ml_manifest import (
	ModelConfigCatBoostManifest,
	ModelConfigLightGBMManifest,
	ModelConfigRidgeManifest,
)
from mt.domain.model.model_name import ModelName, normalize_model_name
from mt.domain.model.model_config_statistical_manifest import ModelConfigEtsManifest

ModelConfigManifest: TypeAlias = (
	ModelConfigEtsManifest
	| ModelConfigRidgeManifest
	| ModelConfigLightGBMManifest
	| ModelConfigCatBoostManifest
	| ModelConfigMlpManifest
	| ModelConfigNBeatsManifest
)

MODEL_CONFIG_CLASS_BY_NAME = {
	ModelName.ETS: ModelConfigEtsManifest,
	ModelName.RIDGE: ModelConfigRidgeManifest,
	ModelName.LIGHTGBM: ModelConfigLightGBMManifest,
	ModelName.CATBOOST: ModelConfigCatBoostManifest,
	ModelName.MLP: ModelConfigMlpManifest,
	ModelName.NBEATS: ModelConfigNBeatsManifest,
}

MODEL_CONFIG_TYPES = tuple(MODEL_CONFIG_CLASS_BY_NAME.values())


def build_model_config(
	model_name: str | ModelName,
	payload: dict[str, Any] | ModelConfigManifest | None,
) -> ModelConfigManifest | None:
	resolved_model_name = normalize_model_name(model_name)
	config_type = MODEL_CONFIG_CLASS_BY_NAME.get(resolved_model_name)
	if config_type is None:
		if payload in (None, {}):
			return None
		if isinstance(payload, dict) and not payload:
			return None
		raise ValueError(f"Model '{resolved_model_name}' does not accept config payload")

	if payload is None:
		return config_type()
	if isinstance(payload, config_type):
		return payload
	if isinstance(payload, MODEL_CONFIG_TYPES):
		raise TypeError()
	if not isinstance(payload, dict):
		raise TypeError()
	return config_type.from_mapping(payload)


def serialize_model_config(config: ModelConfigManifest | None) -> dict[str, Any]:
	if config is None:
		return {}
	return config.to_dict()
