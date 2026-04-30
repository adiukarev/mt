from dataclasses import dataclass


@dataclass(slots=True)
class ObservabilityContext:
	run_key: str

	pipeline_type: str

	artifacts_dir: str

	started_at_utc: str

	execution_mode: str = "local"

	tracking_run_id: str | None = None
	tracking_namespace: str | None = None

	runtime_log_path: str | None = None

	events_path: str | None = None
