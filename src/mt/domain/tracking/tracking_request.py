from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class TrackingRequest:
	pipeline_type: str

	manifest_payload: dict[str, Any]

	params: dict[str, Any]

	tags: dict[str, str] | None = None
