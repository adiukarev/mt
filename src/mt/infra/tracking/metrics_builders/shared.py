import pandas as pd

from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.domain.probabilistic.probabilistic import ProbabilisticColumn, ProbabilisticStatus
from mt.infra.tracking.payload_adapter import add_scalar_metric


def add_stage_metrics(metrics: dict[str, float], stage_timings: list[dict[str, object]]) -> None:
	if not stage_timings:
		return
	frame = pd.DataFrame(stage_timings)
	add_scalar_metric(metrics, "pipeline.stage_record_count", len(frame))


def add_dataset_metrics(metrics: dict[str, float], dataset: object) -> None:
	weekly = getattr(dataset, "weekly", None)
	if isinstance(weekly, pd.DataFrame) and not weekly.empty:
		add_scalar_metric(metrics, "dataset.weekly_rows", len(weekly))
		for column, name in (
			("series_id", "dataset.series_count"),
			("category", "dataset.category_count"),
			("store_id", "dataset.store_count"),
			("state_id", "dataset.state_count"),
			("scenario_name", "dataset.scenario_count"),
		):
			if column in weekly.columns:
				add_scalar_metric(metrics, name, weekly[column].nunique())
		return
	if isinstance(dataset, pd.DataFrame) and not dataset.empty:
		add_scalar_metric(metrics, "dataset.rows", len(dataset))
		for column, name in (
			("series_id", "dataset.series_count"),
			("scenario_name", "dataset.scenario_count"),
		):
			if column in dataset.columns:
				add_scalar_metric(metrics, name, dataset[column].nunique())


def add_prediction_metrics(metrics: dict[str, float], frame: object) -> None:
	if not isinstance(frame, pd.DataFrame) or frame.empty:
		return
	add_scalar_metric(metrics, "predictions.rows", len(frame))
	for column, name in (
		("model_name", "predictions.model_count"),
		("forecast_origin", "predictions.origin_count"),
		("horizon", "predictions.horizon_count"),
		("scenario_name", "predictions.scenario_count"),
	):
		if column in frame.columns:
			add_scalar_metric(metrics, name, frame[column].nunique())
	if ProbabilisticColumn.STATUS in frame.columns:
		available = frame[ProbabilisticColumn.STATUS].astype(str) == ProbabilisticStatus.AVAILABLE
		add_scalar_metric(metrics, "predictions.probabilistic_available_rows", int(available.sum()))
		add_scalar_metric(metrics, "predictions.probabilistic_available_share", float(available.mean()))
	if ProbabilisticColumn.SOURCE in frame.columns:
		for source, count in frame[ProbabilisticColumn.SOURCE].astype(str).value_counts().items():
			add_scalar_metric(metrics, f"predictions.probabilistic_source.{source}.rows", count)


def add_metrics_table(
	metrics: dict[str, float],
	prefix: str,
	frame: object,
	selected_model_name: str | None = None,
) -> None:
	if not isinstance(frame, pd.DataFrame) or frame.empty:
		return
	add_scalar_metric(metrics, f"{prefix}.rows", len(frame))
	numeric_columns = [
		column
		for column in frame.columns
		if column not in {"model_name", "horizon", "scenario_name"}
		and pd.api.types.is_numeric_dtype(frame[column])
	]
	for column in numeric_columns:
		values = pd.to_numeric(frame[column], errors="coerce").dropna()
		if values.empty:
			continue
		add_scalar_metric(metrics, f"{prefix}.best.{column}", values.iloc[0])
		add_scalar_metric(metrics, f"{prefix}.mean.{column}", values.mean())
	if selected_model_name and "model_name" in frame.columns:
		selected = frame[frame["model_name"].astype(str) == selected_model_name]
		if selected.empty:
			return
		for column in numeric_columns:
			value = pd.to_numeric(selected[column], errors="coerce").dropna()
			if not value.empty:
				add_scalar_metric(metrics, f"{prefix}.selected.{column}", value.iloc[0])


def add_frame_rows_metric(metrics: dict[str, float], key: str, frame: object) -> None:
	if not isinstance(frame, pd.DataFrame) or frame.empty:
		return
	add_scalar_metric(metrics, key, len(frame))


def add_mapping_metrics(
	metrics: dict[str, float],
	prefix: str,
	payload: object,
	skip_keys: set[str] | None = None,
) -> None:
	if not isinstance(payload, dict):
		return
	for key, value in payload.items():
		if skip_keys and key in skip_keys:
			continue
		add_scalar_metric(metrics, f"{prefix}.{key}", value)


def resolve_value_path(root: object, path: str) -> object:
	current: object = root
	for part in path.split("."):
		if isinstance(current, dict):
			if part not in current:
				return None
			current = current[part]
			continue
		if not hasattr(current, part):
			return None
		current = getattr(current, part)
	return current


def resolve_selected_model_name(ctx: BasePipelineContext) -> str | None:
	value = getattr(ctx, "selected_model_name", None)
	if isinstance(value, str) and value.strip():
		return value
	return None
