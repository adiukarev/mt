from mt.domain.manifest import ModelManifest
from mt.infra.feature.registry import build_feature_registry


def resolve_model_feature_columns(
	model_manifest: ModelManifest,
	available_feature_columns: list[str],
	aggregation_level: str,
) -> list[str]:
	"""Определить итоговый набор признаков модели"""

	model_name = model_manifest.name
	if model_name == "ets":
		return []

	manifest_feature_columns = _resolve_manifest_feature_columns(
		model_manifest,
		available_feature_columns,
		aggregation_level,
	)
	if model_name != "nbeats":
		return manifest_feature_columns

	dl_manifest = model_manifest.dl_manifest
	if dl_manifest is None:
		raise ValueError()

	history_columns = [
		f"__nbeats_hist_{lag}" for lag in range(1, dl_manifest.history_length + 1)
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
