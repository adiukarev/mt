from typing import Any

from mt.domain.probabilistic.probabilistic_settings import (
	DEFAULT_INTERVAL_LEVELS,
	DEFAULT_PROBABILISTIC_QUANTILES,
)
from mt.domain.tracking.tracking_contract import TrackingFieldSpec, resolve_manifest_tracking_contract
from mt.infra.tracking.params_builders.shared import artifact_basename, nested_value
from mt.infra.tracking.payload_adapter import add_scalar_param, to_mapping


def build_tracking_params(pipeline_type: str, manifest: Any) -> dict[str, Any]:
	payload = to_mapping(manifest)
	params: dict[str, Any] = {"pipeline_type": pipeline_type}
	contract = resolve_manifest_tracking_contract(pipeline_type)
	for spec in contract.param_specs:
		add_scalar_param(
			params,
			spec.key,
			_resolve_tracking_field_value(spec, manifest, payload),
		)
	return params


def build_tracking_tags(pipeline_type: str, manifest: Any) -> dict[str, str]:
	payload = to_mapping(manifest)
	tags: dict[str, str] = {"pipeline_type": pipeline_type}
	contract = resolve_manifest_tracking_contract(pipeline_type)
	for spec in contract.tag_specs:
		value = _resolve_tracking_field_value(spec, manifest, payload)
		if value is None:
			continue
		stringified = _stringify_tag_value(value)
		if stringified is None:
			continue
		tags[spec.key] = stringified
	return tags


def _resolve_tracking_field_value(
	spec: TrackingFieldSpec,
	manifest: Any,
	payload: dict[str, Any],
) -> Any:
	value = _resolve_spec_value(spec, manifest, payload)
	return _apply_transform(value, spec.transform_name)


def _resolve_spec_value(
	spec: TrackingFieldSpec,
	manifest: Any,
	payload: dict[str, Any],
) -> Any:
	if spec.resolver_name is not None:
		return _resolve_named_value(spec.resolver_name, manifest, payload)
	for source_path in spec.source_paths:
		resolved = _resolve_payload_path(payload, source_path)
		if resolved is not None:
			return resolved
	return None


def _resolve_payload_path(payload: dict[str, Any], source_path: str) -> Any:
	return nested_value(payload, *source_path.split("."))


def _resolve_named_value(resolver_name: str, manifest: Any, payload: dict[str, Any]) -> Any:
	if resolver_name == "artifact_basename":
		return artifact_basename(payload)
	if resolver_name == "experiment_feature_manifest_enabled":
		return manifest.build_combined_feature_manifest().enabled
	if resolver_name == "experiment_feature_manifest_feature_set":
		return manifest.build_combined_feature_manifest().feature_set
	if resolver_name == "experiment_feature_manifest_lag_count":
		return len(manifest.build_combined_feature_manifest().lags)
	if resolver_name == "experiment_feature_manifest_rolling_window_count":
		return len(manifest.build_combined_feature_manifest().rolling_windows)
	if resolver_name == "experiment_enabled_model_count":
		return len(getattr(manifest, "enabled_models", []))
	if resolver_name == "experiment_enabled_model_names":
		return list(getattr(manifest, "enabled_model_names", []))
	if resolver_name == "experiment_horizon_range":
		horizon_start = nested_value(payload, "backtest", "horizon_start")
		horizon_end = nested_value(payload, "backtest", "horizon_end")
		if horizon_start is None or horizon_end is None:
			return None
		return f"{horizon_start}..{horizon_end}"
	if resolver_name == "default_probabilistic_quantiles":
		return list(DEFAULT_PROBABILISTIC_QUANTILES)
	if resolver_name == "default_probabilistic_interval_levels":
		return list(DEFAULT_INTERVAL_LEVELS)
	if resolver_name == "synthetic_scenario_names":
		scenarios = payload.get("scenarios")
		if not isinstance(scenarios, list):
			return None
		return [item.get("name") for item in scenarios if isinstance(item, dict) and item.get("name")]
	raise KeyError(f"Unsupported tracking resolver: {resolver_name}")


def _apply_transform(value: Any, transform_name: str | None) -> Any:
	if value is None or transform_name is None:
		return value
	if transform_name == "bool":
		return bool(value)
	if transform_name == "len":
		try:
			return len(value)
		except TypeError:
			return None
	if transform_name == "csv":
		if not isinstance(value, (list, tuple, set)):
			return None
		items = [str(item) for item in value if str(item).strip()]
		return ",".join(items) if items else None
	raise KeyError(f"Unsupported tracking transform: {transform_name}")


def _stringify_tag_value(value: Any) -> str | None:
	if isinstance(value, str):
		resolved = value.strip()
		return resolved or None
	if isinstance(value, bool):
		return str(value).lower()
	if isinstance(value, (int, float)):
		return str(value)
	return None
