from pathlib import Path

from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.forecast.forecast_pipeline_manifest import ForecastModelManifest
from mt.domain.model.registry_selection_manifest import RegistrySelectionManifest
from mt.infra.helper.model_registry import parse_registry_metric_tags


def resolve_reference_model_dir(
	model_manifest: ForecastModelManifest | None,
	dataset_manifest: DatasetManifest | None = None,
	execution_mode: str | None = None,
) -> tuple[Path, dict[str, object]]:
	if model_manifest is None:
		raise ValueError("Forecast model source is required")
	if model_manifest.resolve_source_kind(execution_mode) == "registry":
		return _resolve_registry_reference_model_dir(
			model_manifest=model_manifest,
			dataset_manifest=dataset_manifest,
		)
	model_dir = Path(str(model_manifest.local.model_dir))
	return model_dir, {
		"source_kind": "local",
		"model_dir": str(model_dir),
	}


def _resolve_registry_reference_model_dir(
	model_manifest: ForecastModelManifest,
	dataset_manifest: DatasetManifest | None = None,
) -> tuple[Path, dict[str, object]]:
	import mlflow

	client = mlflow.tracking.MlflowClient()
	registry_source = model_manifest.registry
	if dataset_manifest is None:
		raise ValueError(
			"Forecast registry auto-selection requires dataset manifest metadata"
		)
	selection = RegistrySelectionManifest(
		dataset_kind=str(dataset_manifest.kind),
		aggregation_level=dataset_manifest.aggregation_level,
		target_name=dataset_manifest.target_name,
		dag_id=registry_source.selection.dag_id,
		alias=registry_source.selection.alias,
		metric_name=registry_source.selection.metric_name,
		higher_is_better=registry_source.selection.higher_is_better,
	)
	registry_name, version, ranking_metric = _resolve_best_forecast_registry_model(
		client=client,
		selection=selection,
	)
	model_dir = _download_registry_model_dir(registry_name=registry_name, version=version)
	return model_dir, {
		"source_kind": "registry",
		"source_model_registry_name": registry_name,
		"source_model_registry_version": str(version.version),
		"source_model_registry_alias": selection.alias or _get_version_alias(version),
		"selection_strategy": "best_metric",
		"selection_ranking_metric": ranking_metric,
		"selection_registry_alias": selection.alias,
		"selection_dag_id": getattr(registry_source.selection, "dag_id", None),
	}


def _download_registry_model_dir(
	registry_name: str,
	version: object,
) -> Path:
	import mlflow

	artifact_uri = f"models:/{registry_name}/{version.version}"
	downloaded_root = Path(mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri))
	return _resolve_logged_model_bundle_root(downloaded_root)


def _resolve_best_forecast_registry_model(
	client,
	selection: RegistrySelectionManifest,
) -> tuple[str, object, str]:
	candidates = _collect_forecast_registry_candidates(client, selection)
	if not candidates:
		raise FileNotFoundError("No final model versions matched forecast registry selection")
	reverse = bool(selection.higher_is_better)
	ordered = sorted(
		candidates,
		key=lambda item: (item["metric_value"], item["registry_name"]),
		reverse=reverse,
	)
	selected = ordered[0]
	return (
		str(selected["registry_name"]),
		selected["version"],
		f"{selection.metric_name}:{selected['metric_value']}",
	)


def _collect_forecast_registry_candidates(
	client,
	selection: RegistrySelectionManifest,
) -> list[dict[str, object]]:
	prefix = f"mt_{selection.dataset_kind}_{selection.aggregation_level}_{selection.target_name}"
	candidates: list[dict[str, object]] = []
	for registered_model in _search_registered_models(client):
		registry_name = str(getattr(registered_model, "name", "")).strip()
		if registry_name != prefix:
			continue
		try:
			version = client.get_model_version_by_alias(registry_name, selection.alias)
		except Exception:
			continue
		tags = _get_version_tags(version)
		if not _matches_forecast_registry_selection(tags, selection):
			continue
		metrics = parse_registry_metric_tags(tags)
		metric_value = metrics.get(selection.metric_name)
		if not isinstance(metric_value, (int, float)):
			continue
		candidates.append(
			{
				"registry_name": registry_name,
				"version": version,
				"metric_value": float(metric_value),
			}
		)
	return candidates


def _search_registered_models(client) -> list[object]:
	try:
		return list(client.search_registered_models())
	except TypeError:
		return list(client.search_registered_models(None))


def _matches_forecast_registry_selection(
	tags: dict[str, object],
	selection: RegistrySelectionManifest,
) -> bool:
	expected_pairs = {
		"dataset_kind": selection.dataset_kind,
		"aggregation_level": selection.aggregation_level,
		"target_name": selection.target_name,
		"dag_id": selection.dag_id,
	}
	for key, value in expected_pairs.items():
		if value is None:
			continue
		if str(tags.get(key, "")).strip() != value:
			return False
	return True


def _get_version_tags(version: object) -> dict[str, object]:
	raw_tags = getattr(version, "tags", None)
	if isinstance(raw_tags, dict):
		return dict(raw_tags)
	return {}


def _get_version_alias(version: object) -> str | None:
	alias = getattr(version, "alias", None)
	if isinstance(alias, str) and alias.strip():
		return alias.strip()
	return None


def _resolve_logged_model_bundle_root(downloaded_path: Path) -> Path:
	for candidate in _iter_logged_model_bundle_roots(downloaded_path):
		if (candidate / "artifact_manifest.yaml").exists():
			return candidate
	raise FileNotFoundError(
		f"Model bundle is missing inside downloaded registry artifact: {downloaded_path}"
	)


def _iter_logged_model_bundle_roots(downloaded_path: Path) -> list[Path]:
	candidates: list[Path] = [
		downloaded_path,
		downloaded_path / "artifacts" / "model",
		downloaded_path / "artifacts" / "model_bundle",
		downloaded_path / "model",
		downloaded_path / "model_bundle",
	]
	for child in downloaded_path.iterdir() if downloaded_path.exists() else ():
		if not child.is_dir():
			continue
		candidates.extend(
			[
				child,
				child / "artifacts" / "model",
				child / "artifacts" / "model_bundle",
				child / "model",
				child / "model_bundle",
			]
		)
	unique_candidates: list[Path] = []
	seen: set[Path] = set()
	for candidate in candidates:
		if candidate in seen:
			continue
		unique_candidates.append(candidate)
		seen.add(candidate)
	return unique_candidates
