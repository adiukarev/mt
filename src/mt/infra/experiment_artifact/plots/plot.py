from matplotlib import pyplot as plt

from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.infra.artifact.plot_labels import set_axis_labels, translate_label
from mt.infra.artifact.plot_writer import save_figure


def write_model_ranking_wape_plot(ctx: ExperimentPipelineContext) -> None:
	if ctx.overall_metrics is None or ctx.overall_metrics.empty:
		return

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
	save_figure(fig, ctx.artifacts_paths_map.plot_file("model_ranking_wape.png"))


def write_wape_by_horizon_plot(ctx: ExperimentPipelineContext) -> None:
	if ctx.by_horizon_metrics is None or ctx.by_horizon_metrics.empty:
		return

	_write_metric_by_horizon_plot(ctx, "WAPE", "wape_by_horizon.png")


def write_smape_by_horizon_plot(ctx: ExperimentPipelineContext) -> None:
	if ctx.by_horizon_metrics is None or ctx.by_horizon_metrics.empty:
		return

	_write_metric_by_horizon_plot(ctx, "sMAPE", "smape_by_horizon.png")


def write_bias_by_horizon_plot(ctx: ExperimentPipelineContext) -> None:
	if ctx.by_horizon_metrics is None or ctx.by_horizon_metrics.empty:
		return

	_write_metric_by_horizon_plot(ctx, "Bias", "bias_by_horizon.png")


def write_wape_heatmap_model_horizon_plot(ctx: ExperimentPipelineContext) -> None:
	if ctx.by_horizon_metrics is None or ctx.by_horizon_metrics.empty:
		return

	heatmap = ctx.by_horizon_metrics.pivot(index="model_name", columns="horizon", values="WAPE")
	if heatmap.empty:
		return

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
	save_figure(fig, ctx.artifacts_paths_map.plot_file("wape_heatmap_model_horizon.png"))


def write_segment_model_comparison_plot(ctx: ExperimentPipelineContext) -> None:
	if ctx.evaluation is None or ctx.evaluation.metrics_by_segment.empty:
		return

	pivot = ctx.evaluation.metrics_by_segment.pivot(
		index="segment_label",
		columns="model_name",
		values="WAPE",
	)
	if pivot.empty:
		return

	fig, ax = plt.subplots(figsize=(10, 5))
	pivot.plot(kind="bar", ax=ax, title="WAPE по сегментам и моделям")
	set_axis_labels(ax, xlabel="segment", ylabel="WAPE")
	ax.legend(title=translate_label("model_name"))
	save_figure(fig, ctx.artifacts_paths_map.plot_file("segment_model_comparison.png"))


def write_error_distribution_model_plot(ctx: ExperimentPipelineContext) -> None:
	if (
		ctx.predictions is None
		or ctx.predictions.empty
		or ctx.overall_metrics is None
		or ctx.overall_metrics.empty
	):
		return

	model_name = ctx.overall_metrics.iloc[0]["model_name"]
	predictions = ctx.predictions[ctx.predictions["model_name"] == model_name].copy()
	if predictions.empty:
		return

	predictions["abs_error"] = (predictions["actual"] - predictions["prediction"]).abs()
	fig, ax = plt.subplots(figsize=(9, 4))
	ax.hist(predictions["abs_error"], bins=20)
	set_axis_labels(
		ax,
		title=f"Распределение абсолютной ошибки: {model_name}",
		xlabel="absolute_error",
		ylabel="count",
	)
	save_figure(fig, ctx.artifacts_paths_map.plot_file("error_distribution_model.png"))


def write_coverage_by_horizon_plot(ctx: ExperimentPipelineContext) -> None:
	if ctx.probabilistic_by_horizon_metrics is None or ctx.probabilistic_by_horizon_metrics.empty:
		return
	_write_probabilistic_metric_by_horizon_plot(ctx, "Coverage80", "coverage_by_horizon.png")


def write_interval_width_by_horizon_plot(ctx: ExperimentPipelineContext) -> None:
	if ctx.probabilistic_by_horizon_metrics is None or ctx.probabilistic_by_horizon_metrics.empty:
		return
	_write_probabilistic_metric_by_horizon_plot(ctx, "Width80", "interval_width_by_horizon.png")


def write_wis_by_horizon_plot(ctx: ExperimentPipelineContext) -> None:
	if ctx.probabilistic_by_horizon_metrics is None or ctx.probabilistic_by_horizon_metrics.empty:
		return
	_write_probabilistic_metric_by_horizon_plot(ctx, "WIS", "wis_by_horizon.png")


def write_calibration_curve_plot(ctx: ExperimentPipelineContext) -> None:
	if ctx.evaluation is None or ctx.evaluation.probabilistic_calibration_summary.empty:
		return
	frame = ctx.evaluation.probabilistic_calibration_summary.copy()
	aggregated = frame.groupby(["model_name", "horizon"], as_index=False)["available_errors"].max()
	fig, ax = plt.subplots(figsize=(9, 4))
	for model_name, model_frame in aggregated.groupby("model_name"):
		ax.plot(model_frame["horizon"], model_frame["available_errors"], marker="o", label=model_name)
	set_axis_labels(ax, title="Calibration history by horizon", xlabel="horizon", ylabel="available calibration errors")
	ax.legend(title=translate_label("model_name"))
	save_figure(fig, ctx.artifacts_paths_map.plot_file("calibration_curve.png"))


def write_fan_chart_leader(ctx: ExperimentPipelineContext) -> None:
	if (
		ctx.predictions is None
		or ctx.predictions.empty
		or ctx.overall_metrics is None
		or ctx.overall_metrics.empty
	):
		return
	model_name = str(ctx.overall_metrics.iloc[0]["model_name"])
	frame = ctx.predictions[ctx.predictions["model_name"] == model_name].copy()
	if frame.empty:
		return
	leader_series = str(frame["series_id"].iloc[0])
	series_frame = frame[frame["series_id"] == leader_series].sort_values("target_date")
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(series_frame["target_date"], series_frame["q50"].fillna(series_frame["prediction"]), color="#d1495b", linewidth=2.6, marker="o", label="median")
	if series_frame["lo_95"].notna().any():
		ax.fill_between(series_frame["target_date"], series_frame["lo_95"], series_frame["hi_95"], color="#d1495b", alpha=0.10, label="95% interval")
	if series_frame["lo_80"].notna().any():
		ax.fill_between(series_frame["target_date"], series_frame["lo_80"], series_frame["hi_80"], color="#d1495b", alpha=0.20, label="80% interval")
	ax.plot(series_frame["target_date"], series_frame["actual"], color="#2a9d8f", linewidth=2.0, marker="o", label="actual")
	set_axis_labels(ax, title=f"Leader fan chart: {model_name}", xlabel="target_date", ylabel="sales_units")
	ax.legend(loc="upper left")
	save_figure(fig, ctx.artifacts_paths_map.plot_file("fan_chart_leader.png"))


def _write_metric_by_horizon_plot(
	ctx: ExperimentPipelineContext,
	metric: str,
	filename: str,
) -> None:
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
	save_figure(fig, ctx.artifacts_paths_map.plot_file(filename))


def _write_probabilistic_metric_by_horizon_plot(
	ctx: ExperimentPipelineContext,
	metric: str,
	filename: str,
) -> None:
	fig, ax = plt.subplots(figsize=(9, 4))
	for model_name, frame in ctx.probabilistic_by_horizon_metrics.groupby("model_name"):
		ax.plot(frame["horizon"], frame[metric], marker="o", label=model_name)
	set_axis_labels(
		ax,
		title=f"{translate_label(metric)} по горизонтам прогноза",
		xlabel="horizon",
		ylabel=metric,
	)
	ax.legend(title=translate_label("model_name"))
	save_figure(fig, ctx.artifacts_paths_map.plot_file(filename))
