from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from mt.infra.helper.enum import normalize_enum_by_key


class MonitoringDecisionAction(StrEnum):
	NO_ACTION = "no_action"
	RETRAIN_REQUIRED = "retrain_required"
	MANUAL_REVIEW = "manual_review"
	RUN_EXPERIMENT = "run_experiment"
	PROMOTE_TO_DEV = "promote_to_dev"
	MANUAL_REVIEW_REQUIRED = "manual_review_required"
	HOLD = "hold"


MONITORING_DECISION_ACTION_BY_VALUE = {
	action.value: action for action in MonitoringDecisionAction
}


def normalize_monitoring_decision_action(
	value: str | MonitoringDecisionAction,
) -> MonitoringDecisionAction:
	return normalize_enum_by_key(
		value,
		enum_type=MonitoringDecisionAction,
		by_value=MONITORING_DECISION_ACTION_BY_VALUE,
	)


@dataclass(slots=True)
class MonitoringDecision:
	action: MonitoringDecisionAction
	alert_level: str
	quality_gate_passed: bool
	should_run_experiment: bool
	should_promote: bool
	reasons: list[str] = field(default_factory=list)
	metadata: dict[str, Any] = field(default_factory=dict)

	def __post_init__(self) -> None:
		self.action = normalize_monitoring_decision_action(self.action)
		self.alert_level = str(self.alert_level).strip().lower() or "info"

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
