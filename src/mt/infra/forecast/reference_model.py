from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.feature.feature_manifest import FeatureManifest
from mt.domain.forecast.forecast_pipeline_manifest import ForecastModelManifest
from mt.domain.model.model_artifact import ModelArtifactData
from mt.domain.model.model_config_manifest import ModelConfigManifest, build_model_config
from mt.domain.model.model_name import ModelName
from mt.infra.artifact.text_reader import read_yaml_mapping
from mt.infra.forecast.source_resolver import resolve_reference_model_dir
from mt.infra.model_artifact.model.storage import load_model_artifact
from mt.infra.probabilistic.conformal import ConformalCalibrator


@dataclass(slots=True)
class ReferenceModelConfig:
	model_name: ModelName
	artifact: ModelArtifactData | None
	training_dataset_manifest: DatasetManifest | None
	feature_manifest: FeatureManifest
	config: ModelConfigManifest | None
	source_dir: Path
	source_descriptor: dict[str, object] | None = None


def load_reference_model_config(
	model_manifest: ForecastModelManifest | None = None,
	dataset_manifest: DatasetManifest | None = None,
	execution_mode: str | None = None,
) -> ReferenceModelConfig:
	model_dir, source_descriptor = resolve_reference_model_dir(
		model_manifest=model_manifest,
		dataset_manifest=dataset_manifest,
		execution_mode=execution_mode,
	)
	_read_artifact_manifest(model_dir)
	artifact = _load_model_artifact(model_dir)
	artifact = _with_series_calibration_from_predictions(model_dir, artifact)
	return ReferenceModelConfig(
		model_name=artifact.model_name,
		artifact=artifact,
		training_dataset_manifest=artifact.dataset_manifest,
		feature_manifest=artifact.feature_manifest,
		config=build_model_config(artifact.model_name, artifact.model_config or {}),
		source_dir=model_dir,
		source_descriptor=source_descriptor,
	)


def _read_artifact_manifest(model_dir: Path) -> dict[str, object]:
	artifact_manifest_file = model_dir / "artifact_manifest.yaml"
	if not artifact_manifest_file.exists():
		raise FileNotFoundError(f"Missing artifact manifest: {artifact_manifest_file}")
	return read_yaml_mapping(artifact_manifest_file)


def _load_model_artifact(model_dir: Path) -> ModelArtifactData:
	model_pkl_path = model_dir / "model.pkl"
	try:
		return load_model_artifact(model_pkl_path)
	except Exception as error:
		raise FileNotFoundError(f"Saved model artifact is unavailable: {model_pkl_path}") from error


def _with_series_calibration_from_predictions(
	model_dir: Path,
	artifact: ModelArtifactData,
) -> ModelArtifactData:
	state = dict(artifact.conformal_calibrator_state or {})
	if state.get("absolute_errors_by_series_horizon"):
		return artifact

	predictions_file = model_dir / "raw_predictions.csv"
	if not predictions_file.exists():
		return artifact

	predictions = pd.read_csv(predictions_file)
	if predictions.empty or "series_id" not in predictions.columns:
		return artifact

	calibrator = ConformalCalibrator.from_backtest_predictions(predictions)
	artifact.conformal_calibrator_state = calibrator.serialize()
	metadata = dict(artifact.probabilistic_metadata or {})
	metadata["conformal_scope"] = "by_series_horizon_with_horizon_fallback"
	artifact.probabilistic_metadata = metadata
	return artifact
