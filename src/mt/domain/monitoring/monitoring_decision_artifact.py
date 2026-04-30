from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from mt.infra.artifact.text_reader import read_yaml_mapping
from mt.infra.helper.enum import normalize_enum_by_key


class MonitoringDecisionArtifactStatus(StrEnum):
	NEW = "new"
	CONSUMED = "consumed"
	COMPLETED = "completed"


MONITORING_DECISION_ARTIFACT_STATUS_BY_VALUE = {
	status.value: status for status in MonitoringDecisionArtifactStatus
}


def normalize_monitoring_decision_artifact_status(
	value: str | MonitoringDecisionArtifactStatus,
) -> MonitoringDecisionArtifactStatus:
	return normalize_enum_by_key(
		value,
		enum_type=MonitoringDecisionArtifactStatus,
		by_value=MONITORING_DECISION_ARTIFACT_STATUS_BY_VALUE,
	)


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class MonitoringDecisionArtifact:
	decision_action: str
	should_run_experiment: bool
	quality_gate_passed: bool
	alert_level: str
	reasons: list[str] = field(default_factory=list)
	recommended_alias: str | None = None
	monitoring_metrics: dict[str, float] = field(default_factory=dict)
	source_descriptor: dict[str, Any] = field(default_factory=dict)
	status: MonitoringDecisionArtifactStatus = MonitoringDecisionArtifactStatus.NEW
	created_at_utc: str = field(default_factory=utc_now_iso)
	consumed_at_utc: str | None = None
	completed_at_utc: str | None = None

	def __post_init__(self) -> None:
		self.status = normalize_monitoring_decision_artifact_status(self.status)
		self.alert_level = str(self.alert_level).strip().lower() or "info"
		if self.recommended_alias is not None:
			self.recommended_alias = self.recommended_alias.strip().lower() or None

	def mark_consumed(self) -> None:
		self.status = MonitoringDecisionArtifactStatus.CONSUMED
		self.consumed_at_utc = utc_now_iso()

	def mark_completed(self) -> None:
		self.status = MonitoringDecisionArtifactStatus.COMPLETED
		self.completed_at_utc = utc_now_iso()

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)

	@staticmethod
	def from_dict(data: dict[str, Any]) -> "MonitoringDecisionArtifact":
		return MonitoringDecisionArtifact(
			decision_action=str(data.get("decision_action", "")).strip(),
			should_run_experiment=bool(data.get("should_run_experiment", False)),
			quality_gate_passed=bool(data.get("quality_gate_passed", False)),
			alert_level=str(data.get("alert_level", "info")),
			reasons=[str(item) for item in data.get("reasons", [])],
			recommended_alias=data.get("recommended_alias"),
			monitoring_metrics={
				str(key): float(value)
				for key, value in dict(data.get("monitoring_metrics", {})).items()
			},
			source_descriptor=dict(data.get("source_descriptor", {})),
			status=data.get("status", MonitoringDecisionArtifactStatus.NEW.value),
			created_at_utc=str(data.get("created_at_utc", utc_now_iso())),
			consumed_at_utc=data.get("consumed_at_utc"),
			completed_at_utc=data.get("completed_at_utc"),
		)

	@staticmethod
	def load(path: str | Path) -> "MonitoringDecisionArtifact":
		return MonitoringDecisionArtifact.from_dict(read_yaml_mapping(path))
