from pathlib import Path

import pandas as pd

from mt.domain.forecast.forecast_pipeline_context import ForecastPipelineContext
from mt.domain.model.model_name import ModelName
from mt.domain.probabilistic.probabilistic import ProbabilisticColumn
from mt.infra.artifact.text_writer import write_markdown


def write_forecast_report(ctx: ForecastPipelineContext) -> None:
	if (
		ctx.frame is None
		or ctx.predictions is None
		or ctx.metrics is None
		or ctx.reference_model is None
	):
		raise ValueError()

	lines = _build_forecast_report_lines(
		ctx.reference_model.model_name,
		ctx.predictions,
		ctx.metrics,
		ctx.probabilistic_metrics if ctx.probabilistic_metrics is not None else pd.DataFrame(),
		str((ctx.reference_model.source_descriptor or {}).get("source_model_registry_name") or "")
		or ctx.manifest.model.local.model_dir,
		inference_mode=str(ctx.runtime_metadata.get("forecast_inference_mode") or "n/a"),
		inference_reasons=[
			str(item) for item in ctx.runtime_metadata.get("forecast_inference_reasons", [])
		],
	)
	write_markdown(ctx.artifacts_paths_map.report_file("REPORT.md"), lines)


def _build_forecast_report_lines(
	model_name: ModelName,
	predictions: pd.DataFrame,
	metrics: pd.DataFrame,
	probabilistic_metrics: pd.DataFrame,
	artifact_path: str | Path | None,
	inference_mode: str,
	inference_reasons: list[str],
) -> list[str]:
	lines = [
		"# Forecast Report",
		"",
		f"- Модель: {model_name}",
		f"- Рядов с прогнозом: {predictions['series_id'].nunique()}",
		f"- Forecast rows: {len(predictions)}",
		"",
	]
	if artifact_path is not None:
		lines.append(f"- Источник модели: saved artifact `{artifact_path}`")
	if inference_reasons:
		lines.append(f"- Policy notes: {', '.join(inference_reasons)}")
	if not predictions.empty:
		lines.extend(_build_weekly_forecast_lines(predictions))
	lines.append("")
	return lines


def _build_weekly_forecast_lines(predictions: pd.DataFrame) -> list[str]:
	lines = [
		"",
		"## Прогноз по неделям",
	]
	for series_id, group in predictions.sort_values(["series_id", "horizon"]).groupby("series_id",
	                                                                                  sort=True):
		lines.extend(["", f"### Ряд `{series_id}`"])
		for row in group.itertuples(index=False):
			target_date = pd.Timestamp(row.target_date).date().isoformat()
			q50 = _resolve_forecast_value(row)
			lines.append(
				f"- Неделя {target_date}: ожидаемый прогноз модели — {_format_value(q50)} продаж."
				f" 80% вероятностный интервал: {_format_interval(row.lo_80, row.hi_80)}."
				f" 95% вероятностный интервал: {_format_interval(row.lo_95, row.hi_95)}."
			)
	return lines


def _resolve_forecast_value(row: object) -> float:
	q50 = getattr(row, ProbabilisticColumn.Q50)
	if pd.notna(q50):
		return float(q50)
	return float(getattr(row, "prediction"))


def _format_interval(lower: object, upper: object) -> str:
	if pd.isna(lower) or pd.isna(upper):
		return "недоступен"
	return f"{_format_value(float(lower))}–{_format_value(float(upper))}"


def _format_value(value: float) -> str:
	return f"{value:.1f}"
