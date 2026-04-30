from collections.abc import Sequence
import math
from typing import Any

DEFAULT_FINAL_MODEL_ALIAS = "dev"


def build_final_model_registry_name(
	dataset_kind: str,
	aggregation_level: str,
	target_name: str,
) -> str:
	return f"mt_{dataset_kind}_{aggregation_level}_{target_name}"


def normalize_registry_aliases(aliases: Sequence[str] | None) -> list[str]:
	if aliases is None:
		return []
	normalized: list[str] = []
	seen: set[str] = set()
	for alias in aliases:
		value = alias.strip()
		if not value or value in seen:
			continue
		normalized.append(value)
		seen.add(value)
	return normalized


def resolve_final_model_registry_aliases(aliases: Sequence[str] | None) -> list[str]:
	normalized = normalize_registry_aliases(aliases)
	return normalized or [DEFAULT_FINAL_MODEL_ALIAS]


def build_registry_metric_tags(metrics: dict[str, Any] | None) -> dict[str, str]:
	if not metrics:
		return {}
	tags: dict[str, str] = {}
	model_name = metrics.get("model_name")
	if isinstance(model_name, str) and model_name.strip():
		tags["model_name"] = model_name.strip()
	for metric_name, metric_value in metrics.items():
		if metric_name == "model_name":
			continue
		if isinstance(metric_value, bool):
			continue
		if isinstance(metric_value, (int, float)) and not math.isnan(float(metric_value)):
			tags[f"metric.{metric_name}"] = f"{float(metric_value):.12g}"
	return tags


def parse_registry_metric_tags(tags: dict[str, Any] | None) -> dict[str, Any]:
	if not tags:
		return {}
	metrics: dict[str, Any] = {}
	model_name = tags.get("model_name")
	if isinstance(model_name, str) and model_name.strip():
		metrics["model_name"] = model_name.strip()
	for key, value in tags.items():
		if not key.startswith("metric."):
			continue
		metric_name = key.removeprefix("metric.").strip()
		if not metric_name:
			continue
		try:
			metrics[metric_name] = float(value)
		except (TypeError, ValueError):
			continue
	return metrics


def resolve_registry_model_version(
	client: Any,
	registry_model_name: str,
	registry_model_version: str | None = None,
	registry_model_alias: str | None = None,
) -> Any:
	if registry_model_alias:
		return client.get_model_version_by_alias(registry_model_name, registry_model_alias)
	if registry_model_version:
		return client.get_model_version(name=registry_model_name, version=registry_model_version)
	latest_versions = client.get_latest_versions(registry_model_name)
	if not latest_versions:
		raise FileNotFoundError(f"No versions found in registry model '{registry_model_name}'")
	return latest_versions[-1]
