from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mt.domain.probabilistic.probabilistic_settings import (
	DEFAULT_INTERVAL_LEVELS,
	DEFAULT_PROBABILISTIC_QUANTILES,
)
from mt.domain.probabilistic.probabilistic import ProbabilisticSource, ProbabilisticStatus
from mt.infra.probabilistic.schema import finalize_prediction_frame


@dataclass(slots=True)
class ConformalConfig:
	quantiles: tuple[float, ...] = DEFAULT_PROBABILISTIC_QUANTILES
	interval_levels: tuple[float, ...] = DEFAULT_INTERVAL_LEVELS
	min_history: int = 5
	clip_lower_to_zero: bool = True
	scope: str = "by_horizon"


@dataclass(slots=True)
class ConformalCalibrationSummary:
	horizon: int
	available_errors: int

	probabilistic_status: ProbabilisticStatus
	probabilistic_source: ProbabilisticSource

	radius_80: float | None = None
	radius_95: float | None = None


@dataclass(slots=True)
class ConformalCalibrator:
	config: ConformalConfig = field(default_factory=ConformalConfig)
	absolute_errors_by_horizon: dict[int, list[float]] = field(default_factory=dict)
	absolute_errors_by_series_horizon: dict[str, dict[int, list[float]]] = field(default_factory=dict)
	summary_rows: list[dict[str, object]] = field(default_factory=list)

	def predict(
		self,
		point_predictions: np.ndarray,
		horizon: int,
		series_ids: list[str] | np.ndarray | pd.Series | None = None,
	) -> tuple[pd.DataFrame, ConformalCalibrationSummary]:
		if series_ids is None:
			errors = self.absolute_errors_by_horizon.get(horizon, [])
			return self._predict_with_errors(point_predictions, horizon, errors)

		frames: list[pd.DataFrame] = []
		summaries: list[ConformalCalibrationSummary] = []
		for prediction, series_id in zip(point_predictions, series_ids, strict=False):
			errors = self._resolve_errors(horizon=horizon, series_id=str(series_id))
			frame, summary = self._predict_with_errors(
				np.asarray([float(prediction)], dtype=float),
				horizon,
				errors,
			)
			frames.append(frame)
			summaries.append(summary)
		if not frames:
			return self._point_only_frame(point_predictions, horizon, [])
		available_errors = min((summary.available_errors for summary in summaries), default=0)
		status = (
			ProbabilisticStatus.AVAILABLE
			if all(summary.probabilistic_status == ProbabilisticStatus.AVAILABLE for summary in summaries)
			else ProbabilisticStatus.POINT_ONLY
		)
		source = (
			ProbabilisticSource.CONFORMAL
			if status == ProbabilisticStatus.AVAILABLE
			else ProbabilisticSource.NONE
		)
		return pd.concat(frames, ignore_index=True), ConformalCalibrationSummary(
			horizon=horizon,
			available_errors=available_errors,
			probabilistic_status=status,
			probabilistic_source=source,
		)

	def _resolve_errors(self, horizon: int, series_id: str) -> list[float]:
		series_errors = self.absolute_errors_by_series_horizon.get(series_id, {}).get(horizon, [])
		if len(series_errors) >= self.config.min_history:
			return series_errors
		return self.absolute_errors_by_horizon.get(horizon, [])

	def _predict_with_errors(
		self,
		point_predictions: np.ndarray,
		horizon: int,
		errors: list[float],
	) -> tuple[pd.DataFrame, ConformalCalibrationSummary]:
		if len(errors) < self.config.min_history:
			return self._point_only_frame(point_predictions, horizon, errors)

		radius_80 = _conformal_radius(errors, 0.80)
		radius_95 = _conformal_radius(errors, 0.95)
		frame = pd.DataFrame(
			{
				"prediction": point_predictions,
				"q10": point_predictions - radius_80,
				"q50": point_predictions,
				"q90": point_predictions + radius_80,
				"lo_80": point_predictions - radius_80,
				"hi_80": point_predictions + radius_80,
				"lo_95": point_predictions - radius_95,
				"hi_95": point_predictions + radius_95,
				"probabilistic_source": ProbabilisticSource.CONFORMAL,
				"probabilistic_status": ProbabilisticStatus.AVAILABLE,
			}
		)
		frame, _, _ = finalize_prediction_frame(
			frame,
			clip_lower_to_zero=self.config.clip_lower_to_zero,
		)
		return frame, ConformalCalibrationSummary(
			horizon=horizon,
			available_errors=len(errors),
			probabilistic_status=ProbabilisticStatus.AVAILABLE,
			probabilistic_source=ProbabilisticSource.CONFORMAL,
			radius_80=radius_80,
			radius_95=radius_95,
		)

	def _point_only_frame(
		self,
		point_predictions: np.ndarray,
		horizon: int,
		errors: list[float],
	) -> tuple[pd.DataFrame, ConformalCalibrationSummary]:
		frame = pd.DataFrame(
			{
				"prediction": point_predictions,
				"q10": np.nan,
				"q50": np.nan,
				"q90": np.nan,
				"lo_80": np.nan,
				"hi_80": np.nan,
				"lo_95": np.nan,
				"hi_95": np.nan,
				"probabilistic_source": ProbabilisticSource.NONE,
				"probabilistic_status": ProbabilisticStatus.POINT_ONLY,
			}
		)
		return frame, ConformalCalibrationSummary(
			horizon=horizon,
			available_errors=len(errors),
			probabilistic_status=ProbabilisticStatus.POINT_ONLY,
			probabilistic_source=ProbabilisticSource.NONE,
		)

	def update(self, horizon: int, actual: np.ndarray, prediction: np.ndarray) -> None:
		bucket = self.absolute_errors_by_horizon.setdefault(horizon, [])
		bucket.extend(np.abs(actual - prediction).astype(float).tolist())

	def record_summary(
		self,
		horizon: int,
		forecast_origin: pd.Timestamp,
		summary: ConformalCalibrationSummary,
	) -> None:
		self.summary_rows.append(
			{
				"forecast_origin": forecast_origin,
				"horizon": horizon,
				"available_errors": summary.available_errors,
				"probabilistic_status": summary.probabilistic_status,
				"probabilistic_source": summary.probabilistic_source,
				"radius_80": summary.radius_80,
				"radius_95": summary.radius_95,
			}
		)

	def build_summary_frame(self) -> pd.DataFrame:
		if not self.summary_rows:
			return pd.DataFrame(
				columns=[
					"forecast_origin",
					"horizon",
					"available_errors",
					"probabilistic_status",
					"probabilistic_source",
					"radius_80",
					"radius_95",
				]
			)
		return pd.DataFrame(self.summary_rows).sort_values(["horizon", "forecast_origin"]).reset_index(drop=True)

	def serialize(self) -> dict[str, object]:
		return {
			"config": {
				"quantiles": list(self.config.quantiles),
				"interval_levels": list(self.config.interval_levels),
				"min_history": self.config.min_history,
				"clip_lower_to_zero": self.config.clip_lower_to_zero,
				"scope": self.config.scope,
			},
			"absolute_errors_by_horizon": {
				int(horizon): [float(value) for value in values]
				for horizon, values in self.absolute_errors_by_horizon.items()
			},
			"absolute_errors_by_series_horizon": {
				str(series_id): {
					int(horizon): [float(value) for value in values]
					for horizon, values in horizon_errors.items()
				}
				for series_id, horizon_errors in self.absolute_errors_by_series_horizon.items()
			},
		}

	@classmethod
	def from_serialized(cls, payload: dict[str, object] | None) -> "ConformalCalibrator":
		if not payload:
			return cls()

		if not isinstance(payload, dict):
			raise TypeError()
		config_payload = _mapping_section(payload, "config")
		errors_payload = _mapping_section(payload, "absolute_errors_by_horizon")
		series_errors_payload = _mapping_section(payload, "absolute_errors_by_series_horizon")
		config = ConformalConfig(**config_payload)
		return cls(
			config=config,
			absolute_errors_by_horizon={
				int(horizon): [float(value) for value in values]
				for horizon, values in errors_payload.items()
			},
			absolute_errors_by_series_horizon={
				str(series_id): {
					int(horizon): [float(value) for value in values]
					for horizon, values in dict(series_payload).items()
				}
				for series_id, series_payload in series_errors_payload.items()
				if isinstance(series_payload, dict)
			},
		)

	@classmethod
	def from_backtest_predictions(
		cls,
		predictions: pd.DataFrame,
		config: ConformalConfig | None = None,
	) -> "ConformalCalibrator":
		calibrator = cls(config=config or ConformalConfig())
		if predictions.empty:
			return calibrator

		for horizon, frame in predictions.groupby("horizon", sort=True):
			calibrator.absolute_errors_by_horizon[int(horizon)] = (
				(frame["actual"].astype(float) - frame["prediction"].astype(float)).abs().tolist()
			)
		if "series_id" in predictions.columns:
			for (series_id, horizon), frame in predictions.groupby(["series_id", "horizon"], sort=True):
				series_errors = calibrator.absolute_errors_by_series_horizon.setdefault(str(series_id), {})
				series_errors[int(horizon)] = (
					(frame["actual"].astype(float) - frame["prediction"].astype(float)).abs().tolist()
				)
		return calibrator


def _conformal_radius(errors: list[float], level: float) -> float:
	if not errors:
		return float("nan")
	values = np.asarray(errors, dtype=float)
	return float(np.quantile(values, level, method="higher"))


def _mapping_section(payload: dict[str, object], key: str) -> dict[str, object]:
	value = payload.get(key)
	if value is None:
		return {}
	if not isinstance(value, dict):
		raise TypeError(f"Section '{key}' must be a mapping")
	return dict(value)
