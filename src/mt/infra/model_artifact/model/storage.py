from dataclasses import asdict
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import pandas as pd

from mt.domain.dataset.dataset import DatasetBundle
from mt.domain.model.model_artifact import ModelArtifactData
from mt.domain.model.model_source_mode import (
	MLFLOW_MODEL_SOURCE_MODES,
	ModelSourceMode,
	normalize_model_source_mode,
)
from mt.infra.artifact.binary_writer import write_pickle
from mt.infra.artifact.binary_reader import read_pickle
from mt.infra.artifact.text_writer import write_yaml

_MLFLOW_LOGGED_MODEL_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def build_mlflow_logged_model_name(
	candidate: str,
	*,
	fallback: str = "model",
) -> str:
	"""Build a MLflow-safe logged model key from a human-readable logical path."""

	value = _MLFLOW_LOGGED_MODEL_NAME_PATTERN.sub("_", candidate.strip()).strip("_")
	return value or fallback


def save_model_artifact(
	output_dir: str | Path,
	artifact: ModelArtifactData,
	feature_registry: pd.DataFrame,
	dataset_bundle: DatasetBundle,
) -> Path:
	root = Path(output_dir)
	_write_model_artifact_bundle(
		root=root,
		artifact=artifact,
		feature_registry=feature_registry,
		dataset_bundle=dataset_bundle,
	)
	return root


def save_model_artifact_by_source(
	output_dir: str | Path,
	artifact: ModelArtifactData,
	feature_registry: pd.DataFrame,
	dataset_bundle: DatasetBundle,
	source_mode: str | ModelSourceMode,
	mlflow_run_id: str | None = None,
	registry_model_name: str | None = None,
	registry_tags: dict[str, str] | None = None,
	registry_aliases: list[str] | None = None,
	registry_description: str | None = None,
	artifact_path_override: str | None = None,
) -> Path:
	resolved_source_mode = normalize_model_source_mode(source_mode)
	if resolved_source_mode == ModelSourceMode.LOCAL_ARTIFACTS:
		return save_model_artifact(output_dir, artifact, feature_registry, dataset_bundle)

	if resolved_source_mode not in MLFLOW_MODEL_SOURCE_MODES:
		raise ValueError(f"Unsupported model artifact storage mode: {source_mode}")

	if not mlflow_run_id:
		raise ValueError("mlflow_run_id is required for MLflow-backed model artifact storage")

	registry_name = None
	logical_artifact_name = artifact_path_override or "model"
	if resolved_source_mode == ModelSourceMode.MLFLOW_REGISTRY and not registry_model_name:
		raise ValueError("registry_model_name is required for mlflow_registry mode")
	if registry_model_name:
		registry_name = registry_model_name
	if registry_name is not None:
		logical_artifact_name = artifact_path_override or "best_model"
	artifact_path = build_mlflow_logged_model_name(
		logical_artifact_name,
		fallback="model",
	)

	with TemporaryDirectory(prefix="mt_model_artifact_") as tmp_dir:
		from mt.infra.tracking.backends.mlflow import log_model_storage_artifacts

		staged_root = Path(tmp_dir) / "model"
		_write_model_artifact_bundle(
			root=staged_root,
			artifact=artifact,
			feature_registry=feature_registry,
			dataset_bundle=dataset_bundle,
		)
		log_kwargs: dict[str, object] = {
			"run_id": mlflow_run_id,
			"model_dir": staged_root,
			"artifact_path": artifact_path,
			"registry_name": registry_name,
			"registry_tags": registry_tags,
		}
		if registry_aliases is not None:
			log_kwargs["registry_aliases"] = registry_aliases
		if registry_description is not None:
			log_kwargs["registry_description"] = registry_description
		log_model_storage_artifacts(
			**log_kwargs,
		)

	return Path(output_dir)


def load_model_artifact(path: str | Path) -> ModelArtifactData:
	payload = read_pickle(path)

	if not isinstance(payload, ModelArtifactData):
		raise TypeError()

	return payload


def _write_model_artifact_bundle(
	root: Path,
	artifact: ModelArtifactData,
	feature_registry: pd.DataFrame,
	dataset_bundle: DatasetBundle,
) -> None:
	root.mkdir(parents=True, exist_ok=True)
	write_pickle(root / "model.pkl", artifact)

	write_yaml(
		root / "artifact_manifest.yaml",
		{
			"model_name": artifact.model_name,
			"dataset_manifest": asdict(artifact.dataset_manifest),
			"feature_manifest": asdict(artifact.feature_manifest),
			"feature_columns": artifact.feature_columns,
			"horizons": artifact.horizons,
			"training_aggregation_level": artifact.training_aggregation_level,
			"training_last_week_start": artifact.training_last_week_start,
			"probabilistic_quantiles": artifact.probabilistic_quantiles,
			"interval_levels": artifact.interval_levels,
			"probabilistic_source_by_horizon": artifact.probabilistic_source_by_horizon,
			"has_conformal_calibrator_state": artifact.conformal_calibrator_state is not None,
			"probabilistic_metadata": artifact.probabilistic_metadata or {},
			"model_config": artifact.model_config or {},
		},
	)
