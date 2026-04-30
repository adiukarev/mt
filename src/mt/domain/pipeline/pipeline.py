from abc import ABC, abstractmethod
import os
from pathlib import Path
import re
import time

from mt.domain.orchestration.orchestration_artifact import OrchestrationArtifactPathsMap
from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.artifact.text_writer import write_csv, write_yaml
from mt.infra.artifact.version_controller import set_version_existing_artifact
from mt.infra.helper.str import to_snake_case_without_suffix
from mt.infra.observability.runtime.bootstrap import (
	emit_pipeline_completed,
	initialize_observability
)
import pandas as pd


class BasePipeline(ABC):
	@property
	def name(self) -> str:
		return to_snake_case_without_suffix(self.__class__.__name__, "Pipeline")

	def __init__(self, stages: list[BasePipelineStage]):
		self.stages = stages

	def run(self, *args, **kwargs) -> BasePipelineContext:
		"""Построить контекст, выполнить этапы и вернуть финальное состояние"""

		ctx = self.build_context(*args, **kwargs)
		set_version_existing_artifact(ctx.manifest.runtime.artifacts_dir)
		self._initialize_context_observability(ctx, ctx.manifest)

		pipeline_started_at = time.perf_counter()

		for stage in self.stages:
			stage.run(ctx)

		ctx.pipeline_wall_time_seconds = time.perf_counter() - pipeline_started_at

		finalized_ctx = self.finalize(ctx)
		emit_pipeline_completed(finalized_ctx)

		return finalized_ctx

	def get_stage_names(self) -> list[str]:
		"""Вернуть stage names в порядке выполнения"""

		return [stage.name for stage in self.stages]

	def get_stage(self, stage_name: str) -> BasePipelineStage:
		"""Найти stage по имени"""

		for stage in self.stages:
			if stage.name == stage_name:
				return stage

		raise KeyError(stage_name)

	def run_stage(self, ctx: BasePipelineContext, stage_name: str) -> BasePipelineContext:
		"""Выполнить отдельный stage над уже подготовленным контекстом"""

		self.get_stage(stage_name).run(ctx)

		return ctx

	def _initialize_context_observability(
		self,
		ctx: BasePipelineContext,
		manifest: object,
		tracking_tags: dict[str, str] | None = None,
		tracking_params: dict[str, object] | None = None,
	) -> BasePipelineContext:
		artifacts_dir = getattr(getattr(manifest, "runtime", None), "artifacts_dir", None)
		if not isinstance(artifacts_dir, str):
			return ctx

		execution_mode = os.getenv("MT_EXECUTION_MODE", "local")
		runtime_log_path: Path | None = None
		events_path: Path | None = None
		if execution_mode == "airflow":
			orchestration_paths = OrchestrationArtifactPathsMap.ensure(artifacts_dir)
			runtime_log_path = orchestration_paths.runtime_log
			artifacts_paths_map = getattr(ctx, "artifacts_paths_map", None)
			if artifacts_paths_map is not None and hasattr(artifacts_paths_map, "run_file"):
				events_path = artifacts_paths_map.run_file("events.jsonl")
			else:
				events_path = Path(artifacts_dir) / "events.jsonl"

		return initialize_observability(
			ctx=ctx,
			pipeline_type=self.name,
			manifest=manifest,
			artifacts_dir=artifacts_dir,
			runtime_log_path=runtime_log_path,
			events_path=events_path,
			execution_mode=execution_mode,
			tracking_tags=tracking_tags,
			tracking_params=tracking_params,
		)

	def _persist_run_artifacts(
		self,
		ctx: BasePipelineContext,
		extra_rows: list[dict[str, object]] | None = None,
	) -> None:
		artifacts_paths_map = getattr(ctx, "artifacts_paths_map", None)
		if artifacts_paths_map is None:
			return

		run_dir = getattr(artifacts_paths_map, "run", None) or getattr(artifacts_paths_map, "root",
		                                                               None)
		if run_dir is None:
			return

		run_path = Path(run_dir)
		run_path.mkdir(parents=True, exist_ok=True)

		manifest = getattr(ctx, "manifest", None)
		manifest_payload: dict[str, object] | None = None
		runtime_manifest_payload = ctx.runtime_metadata.get("logical_manifest_payload")
		if isinstance(runtime_manifest_payload, dict):
			manifest_payload = runtime_manifest_payload
		if manifest is not None:
			if manifest_payload is not None:
				pass
			elif hasattr(manifest, "to_dict"):
				manifest_payload = manifest.to_dict()
			elif isinstance(manifest, dict):
				manifest_payload = manifest

		if manifest_payload is not None:
			write_yaml(run_path / "manifest_snapshot.yaml", manifest_payload)

		stage_timings = getattr(ctx, "stage_timings", [])
		pipeline_wall_time_seconds = getattr(ctx, "pipeline_wall_time_seconds", None)
		write_csv(
			run_path / "catalog.csv",
			self._build_run_catalog_frame(stage_timings, pipeline_wall_time_seconds, extra_rows or []),
		)

	def _build_run_catalog_frame(
		self,
		stage_timings: list[dict[str, object]],
		pipeline_wall_time_seconds: float | None,
		extra_rows: list[dict[str, object]],
	) -> pd.DataFrame:
		return pd.DataFrame(
			[
				{
					"record_type": "pipeline",
					"name": self.name,
					"status": "completed",
					"wall_time_seconds": pipeline_wall_time_seconds,
				},
				*[
					{
						"record_type": "stage",
						"name": str(stage_info["stage_name"]),
						"status": str(stage_info["status"]),
						"wall_time_seconds": stage_info["wall_time_seconds"],
					}
					for stage_info in stage_timings
				],
				*extra_rows,
			]
		)

	@abstractmethod
	def build_context(self, *args, **kwargs) -> BasePipelineContext:
		"""Создать начальный контекст пайплайна"""

		...

	@abstractmethod
	def finalize(self, ctx: BasePipelineContext) -> BasePipelineContext:
		"""Сохранить итоговые артефакты и завершить пайплайн"""

		...
