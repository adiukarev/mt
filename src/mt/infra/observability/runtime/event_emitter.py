from datetime import datetime, timezone
from typing import Any

from mt.infra.artifact.text_writer import append_jsonl
from mt.infra.observability.runtime.context_store import get_observability
from mt.infra.tracking.backend_resolver import resolve_tracking_backend_name


def emit_event(event_type: str, payload: dict[str, Any] | None = None) -> None:
	observability = get_observability()
	if observability is None or not observability.events_path:
		return

	record = {
		"timestamp_utc": datetime.now(timezone.utc).isoformat(),
		"event_type": event_type,
		"run_key": observability.run_key,
		"pipeline_type": observability.pipeline_type,
		"tracking_backend": resolve_tracking_backend_name(observability.execution_mode),
		"tracking_run_id": observability.tracking_run_id,
		"execution_mode": observability.execution_mode,
		"payload": payload or {},
	}
	append_jsonl(observability.events_path, record)
