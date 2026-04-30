from datetime import datetime
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from mt.orchestration.pipeline_resolver import resolve_pipeline_definition
from mt.orchestration.runner import (
	execute_pipeline_stage,
	finalize_pipeline_run,
	initialize_pipeline_run,
)


@dataclass(slots=True, frozen=True)
class PipelineDagSettings:
	schedule: str | None = None
	start_date: datetime = datetime(2026, 1, 1)
	catchup: bool = False
	max_active_runs: int | None = 1
	dag_tags: tuple[str, ...] = ()


def build_pipeline_dag(
	dag_id: str,
	pipeline_type: str,
	default_manifest_path: str,
	description: str,
	settings: PipelineDagSettings | None = None,
	trigger_dag_id: str | None = None,
	trigger_conf_builder: Callable[[dict[str, object]], dict[str, object]] | None = None,
):
	from airflow.exceptions import AirflowSkipException
	from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
	from airflow.sdk import DAG, Param, task

	settings = settings or PipelineDagSettings()

	manifest = _manifest_payload(pipeline_type=pipeline_type, path=default_manifest_path)

	with DAG(
		dag_id=dag_id,
		description=description,
		start_date=settings.start_date,
		schedule=settings.schedule,
		catchup=settings.catchup,
		max_active_runs=settings.max_active_runs,
		render_template_as_native_obj=True,
		params={
			"manifest": Param(manifest, type="object"),
		},
		tags=["mt", pipeline_type, *settings.dag_tags],
	) as dag:
		@task(task_id="init_pipeline_run")
		def init_pipeline_run_task(
			default_manifest: dict[str, object],
			manifest_override: dict[str, object] | None = None,
		) -> dict[str, object]:
			return initialize_pipeline_run(
				dag_id=dag_id,
				pipeline_type=pipeline_type,
				manifest_payload=_merge_manifest(default_manifest, manifest_override),
			)

		@task
		def run_stage_task(state: dict[str, object], stage_name: str) -> dict[str, object]:
			return execute_pipeline_stage(state=state, stage_name=stage_name)

		@task(task_id="finalize_pipeline_run")
		def finalize_pipeline_run_task(state: dict[str, object]) -> dict[str, object]:
			return finalize_pipeline_run(state)

		state = init_pipeline_run_task(
			default_manifest="{{ params.manifest }}",
			manifest_override="{{ dag_run.conf.get('manifest') }}",
		)

		for stage_name in resolve_pipeline_definition(
			pipeline_type).pipeline_factory().get_stage_names():
			state = run_stage_task.override(task_id=stage_name)(state=state, stage_name=stage_name)

		finalized_state = finalize_pipeline_run_task(state)

		if trigger_dag_id is not None and trigger_conf_builder is not None:
			@task(task_id=f"build_{trigger_dag_id}_conf")
			def build_trigger_conf_task(state: dict[str, object]) -> dict[str, object]:
				conf = trigger_conf_builder(state)
				if conf is None:
					raise AirflowSkipException(f"Skip downstream trigger for {trigger_dag_id}")
				return conf

			trigger_conf = build_trigger_conf_task(finalized_state)
			trigger_task = TriggerDagRunOperator(
				task_id=f"trigger_{trigger_dag_id}",
				trigger_dag_id=trigger_dag_id,
				conf=trigger_conf,
				wait_for_completion=False,
			)
			trigger_conf >> trigger_task

	return dag


def _manifest_payload(pipeline_type: str, path: str) -> dict[str, object]:
	return resolve_pipeline_definition(pipeline_type).manifest_loader(path).to_dict()


def _merge_manifest(
	default_manifest: dict[str, object],
	manifest_override: dict[str, object] | None,
) -> dict[str, object]:
	if manifest_override is None:
		return deepcopy(default_manifest)

	merged = deepcopy(default_manifest)
	for key, value in manifest_override.items():
		current = merged.get(key)
		if isinstance(current, dict) and isinstance(value, dict):
			merged[key] = _merge_manifest(current, value)
		else:
			merged[key] = deepcopy(value)
	return merged
