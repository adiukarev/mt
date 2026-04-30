from enum import StrEnum


class PointMetricName(StrEnum):
	WAPE = "WAPE"
	SMAPE = "sMAPE"
	MAE = "MAE"
	RMSE = "RMSE"
	BIAS = "Bias"
	MEDIAN_AE = "MedianAE"


class ProbabilisticMetricName(StrEnum):
	PINBALL_Q10 = "Pinball_q10"
	PINBALL_Q50 = "Pinball_q50"
	PINBALL_Q90 = "Pinball_q90"
	MEAN_PINBALL = "MeanPinball"
	COVERAGE_80 = "Coverage80"
	COVERAGE_95 = "Coverage95"
	WIDTH_80 = "Width80"
	WIDTH_95 = "Width95"
	WIS = "WIS"
	PICP_80 = "PICP80"
	PICP_95 = "PICP95"


POINT_METRIC_NAMES: tuple[PointMetricName, ...] = tuple(PointMetricName)
PROBABILISTIC_METRIC_NAMES: tuple[ProbabilisticMetricName, ...] = tuple(ProbabilisticMetricName)

PROBABILISTIC_METRIC_SORT_ORDER: tuple[ProbabilisticMetricName, ...] = (
	ProbabilisticMetricName.WIS,
	ProbabilisticMetricName.MEAN_PINBALL,
)
