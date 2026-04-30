from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mt.domain.tracking.tracking_backend import TrackingBackend
from mt.domain.tracking.tracking_request import TrackingRequest
from mt.domain.tracking.tracking_run import TrackingRunHandle


class NoopTrackingBackend(TrackingBackend):
	name = "noop"

	def start_run(self, request: TrackingRequest) -> TrackingRunHandle:
		return TrackingRunHandle(
			run_id=None,
			run_key=(
				f"{request.pipeline_type}-"
				f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
			),
			started_at_utc=datetime.now(timezone.utc).isoformat(),
			pipeline_type=request.pipeline_type,
			experiment_name=None,
		)
