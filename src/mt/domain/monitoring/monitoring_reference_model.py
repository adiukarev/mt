from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from mt.infra.helper.enum import normalize_enum_by_key


class MonitoringReferenceModelType(StrEnum):
	SEASONAL_NAIVE = "seasonal_naive"
	ROLLING_MEAN = "rolling_mean"


MONITORING_REFERENCE_MODEL_TYPE_BY_VALUE = {
	model_type.value: model_type for model_type in MonitoringReferenceModelType
}


def normalize_monitoring_reference_model_type(
	value: str | MonitoringReferenceModelType,
) -> MonitoringReferenceModelType:
	return normalize_enum_by_key(
		value,
		enum_type=MonitoringReferenceModelType,
		by_value=MONITORING_REFERENCE_MODEL_TYPE_BY_VALUE,
	)


@dataclass(slots=True)
class MonitoringReferenceModelState:
	model_type: MonitoringReferenceModelType
	lookback_weeks: int
	trained_on_rows: int
	trained_until: str | None
	global_baseline: float
	series_baseline: dict[str, float]

	def __post_init__(self) -> None:
		self.model_type = normalize_monitoring_reference_model_type(self.model_type)
		if self.lookback_weeks < 1:
			raise ValueError()

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
