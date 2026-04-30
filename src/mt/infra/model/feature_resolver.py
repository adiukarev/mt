from mt.domain.model.model_config_dl_manifest import ModelConfigNBeatsManifest
from mt.domain.model.model_manifest import ModelManifest
from mt.domain.model.model_family import FEATURELESS_MODEL_NAMES, HISTORY_WINDOW_MODEL_NAMES
from mt.infra.feature.registry_builder import build_feature_registry


def resolve_model_feature_columns(
	model_manifest: ModelManifest,
	available_feature_columns: list[str],
	aggregation_level: str,
) -> list[str]:
	"""Определить итоговый набор признаков модели"""

	model_name = model_manifest.name
	if model_name in FEATURELESS_MODEL_NAMES:
		return []

	manifest_feature_columns = _resolve_manifest_feature_columns(
		model_manifest,
		available_feature_columns,
		aggregation_level,
	)
	if model_name not in HISTORY_WINDOW_MODEL_NAMES:
		return manifest_feature_columns

	if not isinstance(model_manifest.config, ModelConfigNBeatsManifest):
		raise ValueError("N-BEATS requires ModelConfigNBeatsManifest")
	history_columns = [
		f"__nbeats_hist_{lag}" for lag in range(1, model_manifest.config.history_length + 1)
	]
	return history_columns + [
		column for column in manifest_feature_columns if column not in history_columns
	]


def _resolve_manifest_feature_columns(
	model_manifest: ModelManifest,
	available_feature_columns: list[str],
	aggregation_level: str,
) -> list[str]:
	if not model_manifest.features.enabled:
		return []

	model_feature_registry = build_feature_registry(model_manifest.features, aggregation_level)
	enabled_feature_names = {
		str(name)
		for name in model_feature_registry.loc[model_feature_registry["enabled"], "name"].tolist()
	}
	return [column for column in available_feature_columns if column in enabled_feature_names]
