from enum import StrEnum


class ProbabilisticColumn(StrEnum):
	Q10 = "q10"
	Q50 = "q50"
	Q90 = "q90"
	LO_80 = "lo_80"
	HI_80 = "hi_80"
	LO_95 = "lo_95"
	HI_95 = "hi_95"
	SOURCE = "probabilistic_source"
	STATUS = "probabilistic_status"


class ProbabilisticStatus(StrEnum):
	POINT_ONLY = "point_only"
	AVAILABLE = "available"


class ProbabilisticSource(StrEnum):
	NONE = "none"
	CONFORMAL = "conformal"
	NATIVE = "native"


CANONICAL_PROBABILISTIC_COLUMNS: tuple[ProbabilisticColumn, ...] = (
	ProbabilisticColumn.Q10,
	ProbabilisticColumn.Q50,
	ProbabilisticColumn.Q90,
	ProbabilisticColumn.LO_80,
	ProbabilisticColumn.HI_80,
	ProbabilisticColumn.LO_95,
	ProbabilisticColumn.HI_95,
	ProbabilisticColumn.SOURCE,
	ProbabilisticColumn.STATUS,
)

PROBABILISTIC_VALUE_COLUMNS: tuple[ProbabilisticColumn, ...] = (
	ProbabilisticColumn.Q10,
	ProbabilisticColumn.Q50,
	ProbabilisticColumn.Q90,
	ProbabilisticColumn.LO_80,
	ProbabilisticColumn.HI_80,
	ProbabilisticColumn.LO_95,
	ProbabilisticColumn.HI_95,
)

LOWER_BOUND_COLUMNS: tuple[ProbabilisticColumn, ...] = (
	ProbabilisticColumn.Q10,
	ProbabilisticColumn.LO_80,
	ProbabilisticColumn.LO_95,
)

QUANTILE_COLUMNS: tuple[ProbabilisticColumn, ...] = (
	ProbabilisticColumn.Q10,
	ProbabilisticColumn.Q50,
	ProbabilisticColumn.Q90,
)

ORDERED_INTERVAL_STACK_COLUMNS: tuple[ProbabilisticColumn, ...] = (
	ProbabilisticColumn.LO_95,
	ProbabilisticColumn.LO_80,
	ProbabilisticColumn.Q50,
	ProbabilisticColumn.HI_80,
	ProbabilisticColumn.HI_95,
)
