import numpy as np
import pandas as pd

from mt.domain.probabilistic.probabilistic import ProbabilisticColumn, ProbabilisticSource, ProbabilisticStatus
from mt.infra.probabilistic.conformal import (
	ConformalCalibrationSummary,
	ConformalCalibrator,
)


def build_calibration_summary(
	prediction_frame: pd.DataFrame,
	calibrator: ConformalCalibrator,
	horizon: int,
) -> ConformalCalibrationSummary:
	errors = calibrator.absolute_errors_by_horizon.get(horizon, [])

	return ConformalCalibrationSummary(
		horizon=horizon,
		available_errors=len(errors),
		probabilistic_status=(
			ProbabilisticStatus(str(prediction_frame[ProbabilisticColumn.STATUS].iloc[0]))
			if not prediction_frame.empty
			else ProbabilisticStatus.POINT_ONLY
		),
		probabilistic_source=(
			ProbabilisticSource(str(prediction_frame[ProbabilisticColumn.SOURCE].iloc[0]))
			if not prediction_frame.empty
			else ProbabilisticSource.NONE
		),
		radius_80=_radius(errors, 0.80, calibrator.config.min_history),
		radius_95=_radius(errors, 0.95, calibrator.config.min_history),
	)


def _radius(errors: list[float], level: float, min_history: int) -> float | None:
	if len(errors) < min_history:
		return None
	return float(np.quantile(np.asarray(errors, dtype=float), level, method="higher"))
