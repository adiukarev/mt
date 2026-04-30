from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TrackingRunHandle:
	run_id: str | None
	run_key: str

	started_at_utc: str

	pipeline_type: str

	experiment_name: str | None = None
