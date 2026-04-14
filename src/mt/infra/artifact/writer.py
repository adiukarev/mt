from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from mt.infra.artifact.summary import build_comparison_report, build_run_summary
from mt.infra.artifact.experiment_paths import experiment_artifact_relpath
from mt.infra.artifact.plot_labels import set_axis_labels, translate_label
from mt.domain.experiment import ExperimentPipelineContext


def write_csv(path: Path, frame: pd.DataFrame) -> None:
	"""Сохранить CSV-артефакт и создать родительские директории"""

	path.parent.mkdir(parents=True, exist_ok=True)
	frame.to_csv(path, index=False)


def write_markdown(path: str | Path, lines: list[str]) -> None:
	"""Сохранить markdown-строки в стабильном формате"""

	target = Path(path)
	target.parent.mkdir(parents=True, exist_ok=True)
	target.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_reports(ctx: ExperimentPipelineContext) -> None:
	"""Сохранить стандартный комплект markdown-отчетов"""

	comparison_report = build_comparison_report(
		ctx.overall_metrics,
		ctx.comparison.metrics_by_segment if ctx.comparison is not None else pd.DataFrame(),
		ctx.comparison.bootstrap_ci if ctx.comparison is not None else pd.DataFrame(),
		ctx.comparison.rolling_vs_holdout if ctx.comparison is not None else pd.DataFrame(),
	)

	run_summary = build_run_summary(
		ctx.dataset.aggregation_level,
		ctx.feature_manifest.feature_set,
		len(ctx.windows),
		ctx.manifest.runtime.seed,
		ctx.overall_metrics,
		ctx.model_feature_usage_rows,
		ctx.executed_stages,
		ctx.stage_timings,
		ctx.pipeline_wall_time_seconds,
	)

	write_markdown(
		ctx.artifacts_paths_map.root / experiment_artifact_relpath("comparison_report.md"),
		comparison_report
	)
	write_markdown(
		ctx.artifacts_paths_map.root / experiment_artifact_relpath("run_summary.md"),
		run_summary
	)


def write_plots(ctx: ExperimentPipelineContext):
	"""Сохранить стандартные графики сравнения моделей"""

	plots_dir = ctx.artifacts_paths_map.root / "plots"
	plots_dir.mkdir(parents=True, exist_ok=True)

	if not ctx.overall_metrics.empty:
		fig, ax = plt.subplots(figsize=(9, 4))
		ctx.overall_metrics.sort_values("WAPE").plot(
			kind="bar",
			x="model_name",
			y="WAPE",
			legend=False,
			ax=ax,
			title="Ранжирование моделей по WAPE",
		)
		set_axis_labels(ax, xlabel="model_name", ylabel="WAPE")
		_save_figure(fig, plots_dir / "model_ranking_wape.png")

	if ctx.by_horizon_metrics is not None and not ctx.by_horizon_metrics.empty:
		for metric in ("WAPE", "sMAPE", "Bias"):
			fig, ax = plt.subplots(figsize=(9, 4))
			for model_name, frame in ctx.by_horizon_metrics.groupby("model_name"):
				ax.plot(frame["horizon"], frame[metric], marker="o", label=model_name)
			set_axis_labels(
				ax,
				title=f"{translate_label(metric)} по горизонтам прогноза",
				xlabel="horizon",
				ylabel=metric,
			)
			ax.legend(title=translate_label("model_name"))
			_save_figure(fig, plots_dir / f"{metric.lower()}_by_horizon.png")

		heatmap = ctx.by_horizon_metrics.pivot(index="model_name", columns="horizon", values="WAPE")
		if not heatmap.empty:
			fig, ax = plt.subplots(figsize=(9, 4))
			image = ax.imshow(heatmap.values, aspect="auto")
			set_axis_labels(
				ax,
				title="Тепловая карта WAPE по моделям и горизонтам",
				xlabel="horizon",
				ylabel="model_name",
			)
			ax.set_xticks(range(len(heatmap.columns)), labels=heatmap.columns)
			ax.set_yticks(range(len(heatmap.index)), labels=heatmap.index)
			fig.colorbar(image, ax=ax, label=translate_label("WAPE"))
			_save_figure(fig, plots_dir / "wape_heatmap_model_horizon.png")

	if ctx.comparison is not None and not ctx.comparison.metrics_by_segment.empty:
		pivot = ctx.comparison.metrics_by_segment.pivot(
			index="segment_label",
			columns="model_name",
			values="WAPE",
		)
		if not pivot.empty:
			fig, ax = plt.subplots(figsize=(10, 5))
			pivot.plot(kind="bar", ax=ax, title="WAPE по сегментам и моделям")
			set_axis_labels(ax, xlabel="segment", ylabel="WAPE")
			ax.legend(title=translate_label("model_name"))
			_save_figure(fig, plots_dir / "segment_model_comparison.png")

	if ctx.predictions is not None and not ctx.predictions.empty and not ctx.overall_metrics.empty:
		best_model_name = ctx.overall_metrics.iloc[0]["model_name"]
		best_predictions = ctx.predictions[ctx.predictions["model_name"] == best_model_name].copy()
		if not best_predictions.empty:
			best_predictions["abs_error"] = (
				best_predictions["actual"] - best_predictions["prediction"]
			).abs()
			fig, ax = plt.subplots(figsize=(9, 4))
			ax.hist(best_predictions["abs_error"], bins=20)
			set_axis_labels(
				ax,
				title=f"Распределение абсолютной ошибки: {best_model_name}",
				xlabel="absolute_error",
				ylabel="count",
			)
			_save_figure(fig, plots_dir / "error_distribution_best_model.png")


def _save_figure(fig: plt.Figure, path: Path):
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(path)
	plt.close(fig)
