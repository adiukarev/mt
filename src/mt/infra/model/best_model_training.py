from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mt.domain.dataset import DatasetBundle
from mt.domain.manifest import DatasetManifest, ExperimentManifest, FeatureManifest, ModelManifest
from mt.infra.dataset.load import load_dataset
from mt.infra.dataset.prepare import prepare_dataset
from mt.infra.feature.registry import build_feature_registry
from mt.infra.feature.segmentation import segment_series
from mt.infra.feature.supervised_builder import make_supervised_frame
from mt.infra.model.best_model_artifact import BestModelArtifact, save_best_model_artifact
from mt.infra.model.feature_resolution import resolve_model_feature_columns
from mt.infra.model.registry import build_model_adapter
from mt.infra.artifact.logs.best_model import log_best_model_save


@dataclass(slots=True)
class SelectedTrainedModel:
	"""Результат выбора лучшей модели из артефактов сравнения"""

	model_name: str
	comparison_path: Path
	metrics_row: dict[str, Any]


def train_and_save_best_model(
	manifest: ExperimentManifest,
	output_dir: str | Path,
	model_name: str | None = None,
) -> Path:
	"""Обучить финальную модель на всей доступной истории и сохранить артефакт"""

	resolved_model_name = model_name or resolve_best_trained_model(manifest).model_name
	selected_model_manifest = manifest.get_enabled_model(resolved_model_name)
	if manifest.backtest.mode != "direct":
		raise ValueError()

	raw_dataset = load_dataset(manifest.dataset)

	dataset_bundle = prepare_dataset(manifest.dataset, raw_dataset)

	segments = segment_series(dataset_bundle.weekly)

	feature_registry = build_feature_registry(
		selected_model_manifest.features,
		aggregation_level=dataset_bundle.aggregation_level,
	)

	supervised, feature_columns = make_supervised_frame(
		dataset_bundle.weekly,
		segments,
		selected_model_manifest.features,
	)

	return fit_and_save_best_model_from_context(
		manifest=manifest,
		output_dir=output_dir,
		dataset_bundle=dataset_bundle,
		feature_registry=feature_registry,
		supervised=supervised,
		feature_columns=feature_columns,
		model_name=resolved_model_name,
	)


def resolve_best_trained_model(manifest: ExperimentManifest) -> SelectedTrainedModel:
	"""Выбрать лучшую обучаемую модель и вернуть ее вместе с метаданными сравнения"""

	comparison_path = Path(
		manifest.runtime.artifacts_dir) / "comparison" / "overall_model_comparison.csv"
	if not comparison_path.exists():
		raise FileNotFoundError()

	comparison = pd.read_csv(comparison_path)
	if comparison.empty or "model_name" not in comparison.columns:
		raise ValueError()

	selected_model_name = select_best_model_from_metrics(comparison, manifest.enabled_model_names)
	selected_row = comparison.loc[comparison["model_name"].astype(str) == selected_model_name].iloc[0]
	metrics_row = {
		str(column): (
			value.item() if hasattr(value, "item") else value
		)
		for column, value in selected_row.to_dict().items()
	}

	return SelectedTrainedModel(
		model_name=selected_model_name,
		comparison_path=comparison_path,
		metrics_row=metrics_row,
	)


def select_best_model_from_metrics(overall_metrics: pd.DataFrame, allowed_models: list[str]) -> str:
	"""Выбрать лучшую сериализуемую модель из таблицы сравнения"""

	allowed_model_names = set(allowed_models)
	for model_name in overall_metrics["model_name"]:
		name = str(model_name)
		if name in allowed_model_names:
			return name

	raise ValueError()


def fit_and_save_best_model_from_context(
	manifest: ExperimentManifest,
	output_dir: str | Path,
	dataset_bundle: DatasetBundle | None,
	feature_registry: pd.DataFrame | None,
	supervised: pd.DataFrame | None,
	feature_columns: list[str],
	model_name: str | None,
) -> Path:
	"""Обучить и сохранить финальную модель из уже подготовленного pipeline-контекста"""

	if (
		dataset_bundle is None
		or feature_registry is None
		or supervised is None
		or model_name is None
		or manifest.backtest.mode != "direct"
	):
		raise ValueError()

	model_manifest = manifest.get_enabled_model(model_name)

	horizons = list(range(manifest.backtest.horizon_min, manifest.backtest.horizon_max + 1))
	supervised_frame = _ensure_target_columns(supervised, horizons)

	adapters_by_horizon = {}
	artifact_feature_columns: list[str] | None = None

	for horizon in horizons:
		target_column = f"target_h{horizon}"

		adapter = build_model_adapter(
			model_name,
			model_params=model_manifest.adapter_params,
			dl_manifest=model_manifest.dl_manifest,
		)

		prepared_frame = adapter.prepare_frame(supervised_frame)

		model_feature_columns = resolve_model_feature_columns(
			model_manifest,
			feature_columns,
			dataset_bundle.aggregation_level,
		)

		model_feature_columns = adapter.resolve_feature_columns(prepared_frame, model_feature_columns)
		train_frame = prepared_frame.dropna(subset=[target_column]).copy()

		adapter.fit(
			train_frame=train_frame,
			feature_columns=model_feature_columns,
			target_column=target_column,
			horizon=horizon,
			seed=manifest.runtime.seed,
		)

		adapters_by_horizon[horizon] = adapter

		if artifact_feature_columns is None:
			artifact_feature_columns = model_feature_columns
		elif artifact_feature_columns != model_feature_columns:
			raise ValueError()

	artifact = BestModelArtifact(
		model_name=model_name,
		dataset_manifest=_clone_dataset_manifest(manifest.dataset),
		feature_manifest=_clone_feature_manifest(model_manifest.features),
		feature_columns=artifact_feature_columns or [],
		horizons=horizons,
		adapters_by_horizon=adapters_by_horizon,
		training_aggregation_level=dataset_bundle.aggregation_level,
		training_last_week_start=str(dataset_bundle.weekly["week_start"].max().date()),
	)

	root = Path(output_dir)
	save_best_model_artifact(root, artifact, feature_registry, dataset_bundle)

	log_best_model_save(model_name, horizons, root)

	return root


def _ensure_target_columns(supervised: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
	"""Гарантировать наличие target_h* колонок в supervised-таблице"""

	supervised_frame = supervised.copy()

	for horizon in horizons:
		target_column = f"target_h{horizon}"
		if target_column not in supervised_frame.columns:
			supervised_frame[target_column] = supervised_frame.groupby("series_id")["sales_units"].shift(
				-horizon
			)

	return supervised_frame


def _clone_dataset_manifest(manifest: DatasetManifest) -> DatasetManifest:
	"""Скопировать dataset manifest для стабильного сохранения в артефакте"""

	return DatasetManifest(
		path=manifest.path,
		aggregation_level=manifest.aggregation_level,
		target_name=manifest.target_name,
		week_anchor=manifest.week_anchor,
		sample_rows=manifest.sample_rows,
		series_limit=manifest.series_limit,
		include_promo=manifest.include_promo,
		allow_price_features=manifest.allow_price_features,
	)


def _clone_feature_manifest(manifest: FeatureManifest) -> FeatureManifest:
	return FeatureManifest(
		enabled=manifest.enabled,
		feature_set=manifest.feature_set,
		lags=list(manifest.lags),
		rolling_windows=list(manifest.rolling_windows),
		use_calendar=manifest.use_calendar,
		use_category_encodings=manifest.use_category_encodings,
		use_price=manifest.use_price,
		use_promo=manifest.use_promo,
		use_external=manifest.use_external,
	)
