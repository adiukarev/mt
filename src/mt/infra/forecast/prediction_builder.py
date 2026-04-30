import pandas as pd

from mt.domain.model.model_artifact import ModelArtifactData
from mt.domain.probabilistic.probabilistic import ProbabilisticColumn
from mt.domain.series_segmentation.series_segmentation import segment_series
from mt.infra.feature.supervised_builder import build_supervised_frame
from mt.infra.forecast.reference_model import ReferenceModelConfig
from mt.infra.probabilistic.conformal import ConformalCalibrator
from mt.infra.probabilistic.inference import resolve_probabilistic_frame
from mt.infra.probabilistic.schema import (
	build_empty_prediction_frame,
	finalize_prediction_frame,
)


def build_saved_model_predictions(
	frame: pd.DataFrame,
	reference_model: ReferenceModelConfig,
	horizon_weeks: int,
) -> pd.DataFrame:
	if reference_model.artifact is None:
		raise FileNotFoundError(
			f"Saved model artifact is unavailable: {reference_model.source_dir / 'model.pkl'}"
		)
	forecast_rows = run_saved_model_forecast_window(
		forecast_frame=frame.sort_values(["series_id", "week_start"]).reset_index(drop=True),
		horizon_weeks=horizon_weeks,
		artifact=reference_model.artifact,
	)
	validate_saved_model_predictions(forecast_rows, horizon_weeks)
	if forecast_rows.empty:
		return build_empty_prediction_frame()
	return forecast_rows.sort_values(["horizon", "series_id"]).reset_index(drop=True)


def run_saved_model_forecast_window(
	forecast_frame: pd.DataFrame,
	horizon_weeks: int,
	artifact: ModelArtifactData,
) -> pd.DataFrame:
	forecast_origin = _resolve_forecast_origin(forecast_frame)
	supervised = _build_forecast_supervised_frame(forecast_frame, artifact)
	calibrator = ConformalCalibrator.from_serialized(artifact.conformal_calibrator_state)
	rows: list[dict[str, object]] = []

	for horizon in sorted(h for h in artifact.horizons if h <= horizon_weeks):
		adapter = artifact.adapters_by_horizon.get(horizon)
		if adapter is None:
			continue

		target_column = f"target_h{horizon}"
		prepared_frame = adapter.prepare_frame(supervised)
		predict_frame = _select_predict_frame(
			prepared_frame=prepared_frame,
			feature_columns=artifact.feature_columns,
			target_column=target_column,
			forecast_origin=forecast_origin,
			adapter=adapter,
		)
		if predict_frame.empty:
			continue

		point_predictions = adapter.predict(
			predict_frame=predict_frame,
			feature_columns=artifact.feature_columns,
			target_column=target_column,
			horizon=horizon,
		)
		if len(point_predictions) != len(predict_frame):
			raise ValueError("Prediction row count mismatch")

		probabilistic_frame = resolve_probabilistic_frame(
			adapter=adapter,
			predict_frame=predict_frame,
			feature_columns=artifact.feature_columns,
			target_column=target_column,
			horizon=horizon,
			point_predictions=point_predictions.astype(float),
			calibrator=calibrator,
			quantiles=tuple(artifact.probabilistic_quantiles),
		)
		rows.extend(
			_build_prediction_rows(
				artifact=artifact,
				adapter=adapter,
				predict_frame=predict_frame,
				probabilistic_frame=probabilistic_frame,
				point_predictions=point_predictions,
				forecast_origin=forecast_origin,
				horizon=horizon,
				target_column=target_column,
			)
		)

	if not rows:
		return build_empty_prediction_frame()

	frame, _, _ = finalize_prediction_frame(
		pd.DataFrame(rows).sort_values(["horizon", "series_id"]).reset_index(drop=True)
	)
	return frame


def validate_saved_model_predictions(predictions: pd.DataFrame, horizon_weeks: int) -> None:
	if predictions.empty or predictions["prediction"].isna().any():
		raise ValueError("Saved model artifact produced empty or incomplete predictions")

	actual = predictions["actual"].astype(float)
	actual_available = actual.notna()
	nonzero_actual_share = (
		float((actual.loc[actual_available].abs() > 1e-9).mean())
		if actual_available.any()
		else 0.0
	)
	if (
		(predictions["prediction"].astype(float).abs() <= 1e-9).all()
		and nonzero_actual_share >= 0.8
	):
		raise ValueError(
			"Saved model artifact produced degenerate all-zero predictions "
			"for mostly nonzero actuals"
		)

	observed_horizons = {int(value) for value in predictions["horizon"].unique().tolist()}
	expected_horizons = set(range(1, horizon_weeks + 1))
	if not expected_horizons.issubset(observed_horizons):
		raise ValueError(
			"Saved model artifact does not cover requested horizons: "
			f"expected={sorted(expected_horizons)}, observed={sorted(observed_horizons)}"
		)


def prepare_history_weekly(series_frame: pd.DataFrame) -> pd.DataFrame:
	history = series_frame.loc[series_frame["is_history"].astype(bool)].copy()
	if history.empty:
		raise ValueError("History frame is empty")
	return history.loc[:, _weekly_identity_columns(history)]


def prepare_full_weekly(series_frame: pd.DataFrame) -> pd.DataFrame:
	return series_frame.loc[:, _weekly_identity_columns(series_frame)].copy()


def _build_forecast_supervised_frame(
	forecast_frame: pd.DataFrame,
	artifact: ModelArtifactData,
) -> pd.DataFrame:
	history_weekly = prepare_history_weekly(forecast_frame)
	full_weekly = prepare_full_weekly(forecast_frame)
	segments = segment_series(history_weekly)
	supervised, _ = build_supervised_frame(full_weekly, segments, artifact.feature_manifest)

	for horizon in artifact.horizons:
		target_column = f"target_h{horizon}"
		if target_column not in supervised.columns:
			supervised[target_column] = supervised.groupby("series_id")["sales_units"].shift(-horizon)
	return supervised


def _resolve_forecast_origin(forecast_frame: pd.DataFrame) -> pd.Timestamp:
	history_weeks = sorted(pd.to_datetime(
		forecast_frame.loc[forecast_frame["is_history"].astype(bool), "week_start"].unique()
	))
	if not history_weeks:
		raise ValueError("History window is empty")
	return pd.Timestamp(history_weeks[-1])


def _select_predict_frame(
	prepared_frame: pd.DataFrame,
	feature_columns: list[str],
	target_column: str,
	forecast_origin: pd.Timestamp,
	adapter,
) -> pd.DataFrame:
	predict_candidates = prepared_frame.loc[prepared_frame["week_start"] == forecast_origin].copy()
	predict_frame = adapter.select_inference_frame(predict_candidates, feature_columns)
	return predict_frame.copy()


def _build_prediction_rows(
	artifact: ModelArtifactData,
	adapter,
	predict_frame: pd.DataFrame,
	probabilistic_frame: pd.DataFrame,
	point_predictions,
	forecast_origin: pd.Timestamp,
	horizon: int,
	target_column: str,
) -> list[dict[str, object]]:
	target_date = forecast_origin + pd.Timedelta(weeks=horizon)
	rows: list[dict[str, object]] = []
	for row_idx, ((_, row), prediction) in enumerate(
		zip(predict_frame.iterrows(), point_predictions, strict=False)
	):
		actual = row.get(target_column)
		rows.append(
			{
				"model_name": artifact.model_name,
				"model_family": adapter.get_model_info().model_family,
				"series_id": row["series_id"],
				"category": row["category"],
				"segment_label": row.get("segment_label"),
				"forecast_origin": forecast_origin,
				"target_date": target_date,
				"horizon": horizon,
				"actual": float(actual) if pd.notna(actual) else float("nan"),
				"prediction": float(probabilistic_frame.iloc[row_idx]["prediction"])
				if pd.notna(probabilistic_frame.iloc[row_idx]["prediction"])
				else float(prediction),
				ProbabilisticColumn.Q10: probabilistic_frame.iloc[row_idx][ProbabilisticColumn.Q10],
				ProbabilisticColumn.Q50: probabilistic_frame.iloc[row_idx][ProbabilisticColumn.Q50],
				ProbabilisticColumn.Q90: probabilistic_frame.iloc[row_idx][ProbabilisticColumn.Q90],
				ProbabilisticColumn.LO_80: probabilistic_frame.iloc[row_idx][ProbabilisticColumn.LO_80],
				ProbabilisticColumn.HI_80: probabilistic_frame.iloc[row_idx][ProbabilisticColumn.HI_80],
				ProbabilisticColumn.LO_95: probabilistic_frame.iloc[row_idx][ProbabilisticColumn.LO_95],
				ProbabilisticColumn.HI_95: probabilistic_frame.iloc[row_idx][ProbabilisticColumn.HI_95],
				ProbabilisticColumn.SOURCE: probabilistic_frame.iloc[row_idx][
					ProbabilisticColumn.SOURCE
				],
				ProbabilisticColumn.STATUS: probabilistic_frame.iloc[row_idx][
					ProbabilisticColumn.STATUS
				],
			}
		)
	return rows


def _weekly_identity_columns(frame: pd.DataFrame) -> list[str]:
	columns = ["series_id", "category", "week_start", "sales_units"]
	if "sku" in frame.columns:
		columns.append("sku")
	return columns
