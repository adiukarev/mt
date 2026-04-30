from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mt.domain.experiment.experiment_pipeline_manifest import ExperimentPipelineManifest
from mt.domain.model.model_name import ModelName, normalize_model_name


@dataclass(slots=True)
class SelectedTrainedModel:
	model_name: ModelName
	comparison_path: Path
	metrics_row: dict[str, Any]


def select_model(manifest: ExperimentPipelineManifest) -> SelectedTrainedModel:
	evaluation_path = Path(manifest.runtime.artifacts_dir) / "evaluation" / "overall_model_evaluation.csv"
	if not evaluation_path.exists():
		raise FileNotFoundError()

	evaluation = pd.read_csv(evaluation_path)
	if evaluation.empty or "model_name" not in evaluation.columns:
		raise ValueError()

	selected_model_name = select_model_from_metrics(evaluation, manifest.enabled_model_names)
	selected_row = evaluation.loc[evaluation["model_name"].astype(str) == selected_model_name].iloc[0]
	metrics_row = {
		str(column): value.item() if hasattr(value, "item") else value
		for column, value in selected_row.to_dict().items()
	}

	return SelectedTrainedModel(
		model_name=selected_model_name,
		comparison_path=evaluation_path,
		metrics_row=metrics_row,
	)


def select_model_from_metrics(
	overall_metrics: pd.DataFrame,
	allowed_models: list[ModelName],
) -> ModelName:
	allowed_model_names = frozenset(allowed_models)
	for model_name in overall_metrics["model_name"]:
		resolved_model_name = normalize_model_name(str(model_name))
		if resolved_model_name in allowed_model_names:
			return resolved_model_name

	raise ValueError()
