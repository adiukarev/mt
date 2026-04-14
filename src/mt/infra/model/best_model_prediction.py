from pathlib import Path

import pandas as pd

from mt.domain.manifest import DatasetManifest
from mt.infra.dataset.load import load_dataset
from mt.infra.dataset.prepare import prepare_dataset
from mt.infra.feature.segmentation import segment_series
from mt.infra.feature.supervised_builder import make_supervised_frame
from mt.infra.model.best_model_artifact import BestModelArtifact, load_best_model_artifact
from mt.infra.artifact.logs.best_model import log_best_model_prediction


def predict_with_saved_model(
	artifact_path: str | Path,
	dataset_path: str,
	output_path: str | Path | None = None,
) -> pd.DataFrame:
	"""Загрузить финальную модель и построить прогноз на совместимом датасете."""

	artifact = load_best_model_artifact(artifact_path)
	inference_manifest = _build_inference_manifest(artifact, dataset_path)
	raw_dataset = load_dataset(inference_manifest)
	dataset_bundle = prepare_dataset(inference_manifest, raw_dataset)
	segments = segment_series(dataset_bundle.weekly)
	supervised, _ = make_supervised_frame(dataset_bundle.weekly, segments, artifact.feature_manifest)

	forecast_origin = supervised["week_start"].max()
	if pd.isna(forecast_origin):
		raise ValueError()

	rows: list[dict[str, object]] = []
	for horizon in artifact.horizons:
		adapter = artifact.adapters_by_horizon[horizon]
		prepared_frame = adapter.prepare_frame(supervised)
		predict_frame = prepared_frame[prepared_frame["week_start"] == forecast_origin].copy()
		predict_frame = adapter.select_inference_frame(predict_frame, artifact.feature_columns)
		if predict_frame.empty:
			continue

		predictions = adapter.predict(
			predict_frame=predict_frame,
			feature_columns=artifact.feature_columns,
			target_column=f"target_h{horizon}",
			horizon=horizon,
		)
		if len(predictions) != len(predict_frame):
			raise ValueError()

		target_date = pd.Timestamp(forecast_origin) + pd.Timedelta(weeks=horizon)
		for (_, row), prediction in zip(predict_frame.iterrows(), predictions, strict=False):
			rows.append(
				{
					"model_name": artifact.model_name,
					"series_id": row["series_id"],
					"category": row["category"],
					"segment_label": row.get("segment_label"),
					"forecast_origin": forecast_origin,
					"target_date": target_date,
					"horizon": horizon,
					"prediction": float(prediction),
				}
			)

	if not rows:
		result = pd.DataFrame(
			columns=[
				"model_name",
				"series_id",
				"category",
				"segment_label",
				"forecast_origin",
				"target_date",
				"horizon",
				"prediction",
			]
		)
	else:
		result = pd.DataFrame(rows).sort_values(["horizon", "series_id"]).reset_index(drop=True)

	if output_path is not None:
		Path(output_path).parent.mkdir(parents=True, exist_ok=True)
		result.to_csv(output_path, index=False)

		log_best_model_prediction(artifact.model_name, result, output_path)

	return result


def _build_inference_manifest(artifact: BestModelArtifact, dataset_path: str) -> DatasetManifest:
	"""Построить совместимый dataset manifest для боевого инференса"""

	source = artifact.dataset_manifest
	return DatasetManifest(
		path=dataset_path,
		aggregation_level=source.aggregation_level,
		target_name=source.target_name,
		week_anchor=source.week_anchor,
		sample_rows=source.sample_rows,
		series_limit=source.series_limit,
		include_promo=source.include_promo,
		allow_price_features=source.allow_price_features,
	)
