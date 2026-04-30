import pandas as pd

from mt.domain.dataset.dataset import DatasetBundle
from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.model.model_artifact import ModelArtifactData
from mt.domain.model.model_config_manifest import serialize_model_config
from mt.domain.model.model_manifest import ModelManifest
from mt.domain.model.model_name import ModelName
from mt.domain.probabilistic.probabilistic import ProbabilisticSource
from mt.infra.model_artifact.model.builder import build_model_artifact_data
from mt.infra.probabilistic.conformal import ConformalCalibrator


def build_backtest_only_model_artifact(
	*,
	model_name: ModelName,
	model_manifest: ModelManifest,
	dataset_manifest: DatasetManifest,
	dataset_bundle: DatasetBundle,
	horizons: list[int],
	used_feature_columns: list[str],
	backtest_predictions: pd.DataFrame,
	backtest_probabilistic_metadata: dict[str, object] | None,
) -> ModelArtifactData:
	conformal_state = (
		ConformalCalibrator.from_backtest_predictions(backtest_predictions).serialize()
		if not backtest_predictions.empty
		else None
	)
	return build_model_artifact_data(
		model_name=model_name,
		dataset_manifest=dataset_manifest,
		feature_manifest=model_manifest.features,
		feature_columns=list(used_feature_columns),
		horizons=list(horizons),
		adapters_by_horizon={},
		training_aggregation_level=dataset_bundle.aggregation_level,
		training_last_week_start=str(dataset_bundle.weekly["week_start"].max().date()),
		probabilistic_source_by_horizon={h: ProbabilisticSource.CONFORMAL for h in horizons},
		model_config=serialize_model_config(model_manifest.config),
		conformal_calibrator_state=conformal_state,
		probabilistic_metadata=backtest_probabilistic_metadata,
	)
