from typing import Any

from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.feature.feature_manifest import FeatureManifest
from mt.domain.model.model_artifact import ModelArtifactData
from mt.domain.model.model_name import ModelName
from mt.domain.probabilistic.probabilistic import ProbabilisticSource
from mt.domain.probabilistic.probabilistic_settings import (
	DEFAULT_INTERVAL_LEVELS,
	DEFAULT_PROBABILISTIC_QUANTILES,
)


def build_model_artifact_data(
	*,
	model_name: ModelName,
	dataset_manifest: DatasetManifest,
	feature_manifest: FeatureManifest,
	feature_columns: list[str],
	horizons: list[int],
	adapters_by_horizon: dict[int, Any],
	training_aggregation_level: str,
	training_last_week_start: str,
	probabilistic_source_by_horizon: dict[int, ProbabilisticSource],
	model_config: dict[str, Any] | None,
	conformal_calibrator_state: dict[str, Any] | None,
	probabilistic_metadata: dict[str, Any] | None,
	probabilistic_quantiles: tuple[float, ...] = DEFAULT_PROBABILISTIC_QUANTILES,
	interval_levels: tuple[float, ...] = DEFAULT_INTERVAL_LEVELS,
) -> ModelArtifactData:
	return ModelArtifactData(
		model_name=model_name,
		dataset_manifest=dataset_manifest,
		feature_manifest=feature_manifest,
		feature_columns=list(feature_columns),
		horizons=list(horizons),
		adapters_by_horizon=adapters_by_horizon,
		training_aggregation_level=training_aggregation_level,
		training_last_week_start=training_last_week_start,
		probabilistic_quantiles=list(probabilistic_quantiles),
		interval_levels=list(interval_levels),
		probabilistic_source_by_horizon=probabilistic_source_by_horizon,
		model_config=model_config,
		conformal_calibrator_state=conformal_calibrator_state,
		probabilistic_metadata=dict(probabilistic_metadata or {}),
	)
