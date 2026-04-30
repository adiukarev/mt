import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from mt.domain.dataset.dataset import DatasetBundle
from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.experiment.experiment_pipeline_manifest import ExperimentPipelineManifest
from mt.domain.feature.feature_manifest import FeatureManifest
from mt.domain.model.model_artifact import ModelArtifactData
from mt.domain.model.model_manifest import ModelManifest
from mt.domain.model.model_config_manifest import serialize_model_config
from mt.domain.model.model_family import ModelFamily
from mt.infra.observability.runtime.stage_events import (
	log_horizon_start,
	log_model_runner_end,
	log_model_runner_start,
	log_model_save,
)
from mt.domain.model.model_name import ModelName, normalize_model_name
from mt.domain.model.model_source_mode import ModelSourceMode
from mt.domain.probabilistic.probabilistic import ProbabilisticSource
from mt.infra.dataset.loader import load_dataset
from mt.infra.dataset.preparation import prepare_dataset
from mt.infra.feature.registry_builder import build_feature_registry
from mt.domain.series_segmentation.series_segmentation import segment_series
from mt.infra.feature.supervised_builder import build_supervised_frame
from mt.infra.backtest.windows_builder import build_backtest_windows
from mt.infra.model.feature_resolver import resolve_model_feature_columns
from mt.infra.model.selector import select_model
from mt.infra.model.adapter_builder import build_model_adapter
from mt.infra.backtest.runner import run_backtest
from mt.infra.model_artifact.model.builder import build_model_artifact_data
from mt.infra.model_artifact.model.storage import save_model_artifact_by_source
from mt.infra.probabilistic.conformal import ConformalCalibrator


@dataclass(slots=True)
class TrainedModelBundle:
	artifact: ModelArtifactData
	feature_registry: pd.DataFrame
	dataset_bundle: DatasetBundle


def train_and_save_model(
	manifest: ExperimentPipelineManifest,
	output_dir: str | Path,
	model_name: str | ModelName | None = None,
) -> Path:
	trained_bundle = train_model(manifest=manifest, model_name=model_name)
	return save_trained_model_bundle(output_dir=output_dir, trained_bundle=trained_bundle)


def train_model(
	manifest: ExperimentPipelineManifest,
	model_name: str | ModelName | None = None,
) -> TrainedModelBundle:
	resolved_model_name = (
		normalize_model_name(model_name)
		if model_name is not None
		else select_model(manifest).model_name
	)
	selected_model_manifest = manifest.get_enabled_model(resolved_model_name)

	raw_dataset = load_dataset(manifest.dataset)
	dataset_bundle = prepare_dataset(manifest.dataset, raw_dataset)
	segments = segment_series(dataset_bundle.weekly)
	feature_registry = build_feature_registry(
		selected_model_manifest.features,
		aggregation_level=dataset_bundle.aggregation_level,
	)
	supervised, feature_columns = build_supervised_frame(
		dataset_bundle.weekly,
		segments,
		selected_model_manifest.features,
		manifest.backtest,
	)

	return fit_model_bundle_from_context(
		manifest=manifest,
		dataset_bundle=dataset_bundle,
		feature_registry=feature_registry,
		supervised=supervised,
		feature_columns=feature_columns,
		model_name=resolved_model_name,
	)


def fit_model_bundle_from_context(
	manifest: ExperimentPipelineManifest,
	dataset_bundle: DatasetBundle | None,
	feature_registry: pd.DataFrame | None,
	supervised: pd.DataFrame | None,
	feature_columns: list[str],
	model_name: ModelName | None,
	model_artifact_dir: str | Path | None = None,
	backtest_predictions: pd.DataFrame | None = None,
	backtest_probabilistic_metadata: dict[str, object] | None = None,
	source_artifact: ModelArtifactData | None = None,
) -> TrainedModelBundle:
	if (
		dataset_bundle is None
		or feature_registry is None
		or supervised is None
		or model_name is None
	):
		raise ValueError()

	model_manifest = manifest.get_enabled_model(model_name)
	horizons = list(range(manifest.backtest.horizon_start, manifest.backtest.horizon_end + 1))
	final_forecast_origin = _resolve_final_forecast_origin(
		manifest=manifest,
		dataset_bundle=dataset_bundle,
	)
	model_artifact_root = (
		Path(model_artifact_dir)
		if model_artifact_dir is not None
		else Path(manifest.runtime.artifacts_dir) / "model"
	)

	conformal_calibrator, probabilistic_metadata = _resolve_calibrator_for_refit(
		manifest=manifest,
		model_name=model_name,
		model_manifest=model_manifest,
		dataset_bundle=dataset_bundle,
		supervised=supervised,
		feature_columns=feature_columns,
		model_artifact_root=model_artifact_root,
		backtest_predictions=backtest_predictions,
		backtest_probabilistic_metadata=backtest_probabilistic_metadata,
		source_artifact=source_artifact,
	)

	adapters_by_horizon = {}
	artifact_feature_columns: list[str] | None = None
	probabilistic_source_by_horizon: dict[int, ProbabilisticSource] = {}

	for horizon in horizons:
		log_horizon_start(model_name=model_name, horizon=horizon, windows=1)
		target_column = f"target_h{horizon}"
		adapter = _build_adapter_for_refit(
			model_name=model_name,
			model_manifest=model_manifest,
			source_artifact=source_artifact,
			horizon=horizon,
		)
		prepared_frame = adapter.prepare_frame(supervised)
		model_feature_columns = resolve_model_feature_columns(
			model_manifest,
			feature_columns,
			dataset_bundle.aggregation_level,
		)
		model_feature_columns = adapter.resolve_feature_columns(prepared_frame, model_feature_columns)
		train_end = (
			final_forecast_origin - pd.Timedelta(weeks=horizon)
			if adapter.get_model_info().model_family == ModelFamily.ML
			else final_forecast_origin
		)
		train_frame = prepared_frame.loc[
			pd.to_datetime(prepared_frame["week_start"]) <= train_end
		].copy()
		if adapter.get_model_info().model_family == ModelFamily.ML:
			train_frame = train_frame.dropna(subset=[target_column]).copy()
		else:
			train_frame = train_frame.dropna(subset=["sales_units"]).copy()
		if train_frame.empty:
			raise ValueError(
				"Final model training frame is empty: "
				f"model={model_name}, horizon={horizon}, train_end={train_end.date()}"
			)
		adapter.fit(
			train_frame=train_frame,
			feature_columns=model_feature_columns,
			target_column=target_column,
			horizon=horizon,
			seed=manifest.runtime.seed,
		)
		adapters_by_horizon[horizon] = adapter
		probabilistic_source_by_horizon[horizon] = (
			ProbabilisticSource.NATIVE
			if adapter.supports_native_probabilistic()
			else ProbabilisticSource.CONFORMAL
		)

		if artifact_feature_columns is None:
			artifact_feature_columns = model_feature_columns
		elif artifact_feature_columns != model_feature_columns:
			raise ValueError()

	artifact = build_model_artifact_data(
		model_name=model_name,
		dataset_manifest=_clone_dataset_manifest(manifest.dataset),
		feature_manifest=_clone_feature_manifest(model_manifest.features),
		model_config=serialize_model_config(model_manifest.config),
		feature_columns=artifact_feature_columns or [],
		horizons=horizons,
		adapters_by_horizon=adapters_by_horizon,
		training_aggregation_level=dataset_bundle.aggregation_level,
		training_last_week_start=str(final_forecast_origin.date()),
		probabilistic_source_by_horizon=probabilistic_source_by_horizon,
		conformal_calibrator_state=conformal_calibrator.serialize(),
		probabilistic_metadata=probabilistic_metadata,
	)

	return TrainedModelBundle(
		artifact=artifact,
		feature_registry=feature_registry,
		dataset_bundle=dataset_bundle,
	)


def save_trained_model_bundle(
	output_dir: str | Path,
	trained_bundle: TrainedModelBundle,
	source_mode: str | ModelSourceMode = ModelSourceMode.LOCAL_ARTIFACTS,
	mlflow_run_id: str | None = None,
	registry_model_name: str | None = None,
	registry_tags: dict[str, str] | None = None,
) -> Path:
	root = Path(output_dir)
	save_model_artifact_by_source(
		root,
		trained_bundle.artifact,
		trained_bundle.feature_registry,
		trained_bundle.dataset_bundle,
		source_mode=source_mode,
		mlflow_run_id=mlflow_run_id,
		registry_model_name=registry_model_name,
		registry_tags=registry_tags,
	)
	log_model_save(
		trained_bundle.artifact.model_name,
		trained_bundle.artifact.horizons,
		root,
	)
	return root


def _resolve_final_forecast_origin(
	manifest: ExperimentPipelineManifest,
	dataset_bundle: DatasetBundle,
) -> pd.Timestamp:
	holdout_tail_weeks = manifest.backtest.resolve_holdout_tail_weeks()
	max_week_start = pd.Timestamp(dataset_bundle.weekly["week_start"].max())
	if holdout_tail_weeks == 0:
		return max_week_start
	return max_week_start - pd.Timedelta(weeks=holdout_tail_weeks)


def _build_adapter_for_refit(
	model_name: ModelName,
	model_manifest: ModelManifest,
	source_artifact: ModelArtifactData | None,
	horizon: int,
):
	if source_artifact is not None and source_artifact.model_name == model_name:
		source_adapter = source_artifact.adapters_by_horizon.get(horizon)
		if source_adapter is not None:
			return copy.deepcopy(source_adapter)
	return build_model_adapter(model_name, model_manifest.config)


def _resolve_calibrator_for_refit(
	manifest: ExperimentPipelineManifest,
	model_name: ModelName,
	model_manifest: ModelManifest,
	dataset_bundle: DatasetBundle,
	supervised: pd.DataFrame,
	feature_columns: list[str],
	model_artifact_root: Path,
	backtest_predictions: pd.DataFrame | None,
	backtest_probabilistic_metadata: dict[str, object] | None,
	source_artifact: ModelArtifactData | None,
) -> tuple[ConformalCalibrator, dict[str, object] | None]:
	if backtest_predictions is not None:
		return (
			ConformalCalibrator.from_backtest_predictions(backtest_predictions),
			dict(backtest_probabilistic_metadata or {}),
		)
	if source_artifact is not None and source_artifact.conformal_calibrator_state is not None:
		return (
			ConformalCalibrator.from_serialized(source_artifact.conformal_calibrator_state),
			dict(source_artifact.probabilistic_metadata or {}),
		)

	backtest_windows = pd.DataFrame(
		asdict(window) for window in build_backtest_windows(manifest.backtest, dataset_bundle.weekly)
	)
	log_model_runner_start(model_name, backtest_windows)
	backtest_result = run_backtest(
		model_name=model_name,
		supervised=supervised,
		feature_columns=feature_columns,
		windows=backtest_windows,
		seed=manifest.runtime.seed,
		config=model_manifest.config,
	)
	log_model_runner_end(backtest_result, model_artifact_root)
	return (
		ConformalCalibrator.from_backtest_predictions(backtest_result.predictions),
		dict(backtest_result.probabilistic_metadata or {}),
	)


def _clone_dataset_manifest(manifest: DatasetManifest) -> DatasetManifest:
	return DatasetManifest(
		kind=manifest.kind,
		path=manifest.path,
		aggregation_level=manifest.aggregation_level,
		target_name=manifest.target_name,
		week_anchor=manifest.week_anchor,
		series_limit=manifest.series_limit,
		series_allowlist=list(
			manifest.series_allowlist) if manifest.series_allowlist is not None else None,
	)


def _clone_feature_manifest(manifest: FeatureManifest) -> FeatureManifest:
	return FeatureManifest(
		enabled=manifest.enabled,
		feature_set=manifest.feature_set,
		lags=list(manifest.lags),
		rolling_windows=list(manifest.rolling_windows),
		use_calendar=manifest.use_calendar,
		use_category_encodings=manifest.use_category_encodings,
	)
