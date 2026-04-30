from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.domain.model.model_result import ModelResult
from mt.infra.helper.model_registry import (
	build_final_model_registry_name,
	build_registry_metric_tags,
	resolve_final_model_registry_aliases,
)


def resolve_experiment_model_registry_name(
	ctx: ExperimentPipelineContext,
) -> str:
	dataset = ctx.require_dataset()
	return build_final_model_registry_name(
		dataset_kind=dataset.kind,
		aggregation_level=dataset.aggregation_level,
		target_name=dataset.target_name,
	)


def build_experiment_model_registry_tags(
	ctx: ExperimentPipelineContext,
	result: ModelResult,
	selected_metrics: dict[str, object],
) -> dict[str, str]:
	dataset = ctx.require_dataset()
	dag_id = _resolve_airflow_dag_id(ctx)
	tags = {
		"run_key": str(ctx.runtime_metadata.get("tracking_run_key", "")),
		"dataset_kind": dataset.kind,
		"aggregation_level": dataset.aggregation_level,
		"target_name": dataset.target_name,
		"model_name": result.info.model_name,
		"model_family": result.info.model_family,
		"horizon_start": str(ctx.manifest.backtest.horizon_start),
		"horizon_end": str(ctx.manifest.backtest.horizon_end),
		"source_pipeline": "experiment",
		"artifact_kind": "model_bundle",
	}
	if dag_id is not None:
		tags["dag_id"] = dag_id
	tags.update(_build_training_series_tags(ctx))
	tags.update(build_registry_metric_tags(selected_metrics))
	return tags


def build_experiment_model_registry_aliases() -> list[str]:
	return resolve_final_model_registry_aliases(None)


def build_experiment_model_registry_description(
	ctx: ExperimentPipelineContext,
	result: ModelResult,
	registry_name: str,
	selected_metrics: dict[str, object],
) -> str:
	dataset = ctx.require_dataset()
	lines = [
		f"Final experiment model bundle: {registry_name}",
		f"Dataset: {dataset.kind}",
		f"Aggregation level: {dataset.aggregation_level}",
		f"Target: {dataset.target_name}",
		f"Model: {result.info.model_name} ({result.info.model_family})",
		f"Backtest horizon: {ctx.manifest.backtest.horizon_start}..{ctx.manifest.backtest.horizon_end}",
		f"Run key: {ctx.runtime_metadata.get('tracking_run_key') or '-'}",
		f"Source pipeline: experiment",
	]
	dag_id = _resolve_airflow_dag_id(ctx)
	if dag_id is not None:
		lines.append(f"Airflow DAG: {dag_id}")
	training_series_summary = _build_training_series_summary(ctx)
	if training_series_summary:
		lines.append(training_series_summary)
	metric_lines = [
		f"{metric_name}: {metric_value}"
		for metric_name, metric_value in selected_metrics.items()
		if metric_name != "model_name"
	]
	if metric_lines:
		lines.append("Metrics:")
		lines.extend(metric_lines)
	return "\n".join(lines)


def _build_training_series_tags(ctx: ExperimentPipelineContext) -> dict[str, str]:
	series_ids = _resolve_training_series_ids(ctx)
	if not series_ids:
		return {}
	tags = {
		"training_series_count": str(len(series_ids)),
		"training_series_ids": _render_training_series_ids(series_ids),
	}
	if ctx.manifest.dataset.series_limit is not None:
		tags["dataset_series_limit"] = str(ctx.manifest.dataset.series_limit)
	if ctx.manifest.dataset.series_allowlist:
		tags["dataset_series_allowlist"] = ", ".join(
			str(item) for item in ctx.manifest.dataset.series_allowlist
		)
	return tags


def _build_training_series_summary(ctx: ExperimentPipelineContext) -> str | None:
	series_ids = _resolve_training_series_ids(ctx)
	if not series_ids:
		return None
	return f"Training series ({len(series_ids)}): {_render_training_series_ids(series_ids)}"


def _resolve_training_series_ids(ctx: ExperimentPipelineContext) -> list[str]:
	dataset = ctx.require_dataset()
	if dataset.weekly is None or "series_id" not in dataset.weekly.columns:
		return []
	return sorted({str(value) for value in dataset.weekly["series_id"].dropna().astype(str)})


def _resolve_airflow_dag_id(ctx: ExperimentPipelineContext) -> str | None:
	value = ctx.runtime_metadata.get("airflow_dag_id")
	if not isinstance(value, str):
		return None
	resolved = value.strip()
	return resolved or None


def _render_training_series_ids(series_ids: list[str], max_items: int = 20, max_chars: int = 500) -> str:
	if len(series_ids) <= max_items:
		rendered = ", ".join(series_ids)
	else:
		head = ", ".join(series_ids[:max_items])
		rendered = f"{head}, ... (+{len(series_ids) - max_items} more)"
	if len(rendered) <= max_chars:
		return rendered
	return rendered[: max_chars - 3].rstrip(", ") + "..."
