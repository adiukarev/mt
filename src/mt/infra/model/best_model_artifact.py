import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mt.domain.dataset import DatasetBundle
from mt.infra.artifact.serialization import dump_json, dump_yaml
from mt.domain.model import BestModelArtifact


def save_best_model_artifact(
	output_dir: str | Path,
	artifact: BestModelArtifact,
	feature_registry: pd.DataFrame,
	dataset_bundle: DatasetBundle,
) -> None:
	"""Сохранить лучшую модель и сопутствующую спецификацию на диск"""

	root = Path(output_dir)
	root.mkdir(parents=True, exist_ok=True)

	with (root / "model.pkl").open("wb") as file_obj:
		pickle.dump(artifact, file_obj)

	dump_yaml(
		root / "artifact_manifest.yaml",
		{
			"model_name": artifact.model_name,
			"dataset_manifest": asdict(artifact.dataset_manifest),
			"feature_manifest": asdict(artifact.feature_manifest),
			"feature_columns": artifact.feature_columns,
			"horizons": artifact.horizons,
			"training_aggregation_level": artifact.training_aggregation_level,
			"training_last_week_start": artifact.training_last_week_start,
		},
	)

	dump_json(
		root / "dataset_metadata.json",
		{
			"aggregation_level": dataset_bundle.aggregation_level,
			"target_name": dataset_bundle.target_name,
			"metadata": _json_safe_mapping(dataset_bundle.metadata),
		},
	)

	feature_registry.to_csv(root / "feature_registry.csv", index=False)


def load_best_model_artifact(path: str | Path) -> BestModelArtifact:
	"""Загрузить финальный модельный артефакт с диска"""

	with Path(path).open("rb") as file_obj:
		payload = pickle.load(file_obj)

	if not isinstance(payload, BestModelArtifact):
		raise TypeError()

	return payload


def _json_safe_mapping(metadata: dict[str, Any]) -> dict[str, Any]:
	"""Привести произвольные метаданные к JSON-совместимому виду"""

	result: dict[str, Any] = {}
	for key, value in metadata.items():
		if isinstance(value, (str, int, float, bool)) or value is None:
			result[key] = value
		else:
			result[key] = str(value)
	return result
