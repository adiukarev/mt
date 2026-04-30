from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from mt.domain.model.model_family import ModelFamily
from mt.domain.model.model_name import ModelName
from mt.domain.tracking.tracking_request import TrackingRequest
from mt.domain.tracking.tracking_run import TrackingRunHandle


class TrackingBackend(Protocol):
	name: str

	def start_run(self, request: TrackingRequest) -> TrackingRunHandle:
		...

	def log_stage_run(
		self,
		run_id: str | None,
		stage_name: str,
		stage_timing: dict[str, object] | None,
		artifact_paths: list[Path],
		artifact_root: str | Path,
	) -> None:
		return None

	def log_pipeline_summary(
		self,
		run_id: str | None,
		metrics: dict[str, float],
		artifact_paths: list[Path],
		artifact_root: str | Path,
	) -> None:
		return None

	def log_experiment_model_run(
		self,
		parent_run_id: str | None,
		model_name: ModelName,
		model_family: ModelFamily,
		model_dir: str | Path,
		model_manifest: dict[str, Any],
		metrics_overall: pd.DataFrame,
		metrics_by_horizon: pd.DataFrame,
		probabilistic_metrics_overall: pd.DataFrame,
		probabilistic_metrics_by_horizon: pd.DataFrame,
		used_feature_columns: list[str],
		calibration_summary: pd.DataFrame | None,
		probabilistic_metadata: dict[str, Any] | None,
		train_time_seconds: float | None,
		inference_time_seconds: float | None,
		wall_time_seconds: float | None,
	) -> None:
		return None

	def log_experiment_model(
		self,
		parent_run_id: str | None,
		model_name: ModelName,
		model_dir: str | Path,
		selected_metrics: dict[str, Any],
		registry_name: str | None = None,
		registry_tags: dict[str, str] | None = None,
		registry_aliases: list[str] | None = None,
		registry_description: str | None = None,
	) -> None:
		return None
