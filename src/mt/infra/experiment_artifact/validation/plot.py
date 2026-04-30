from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd

from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.infra.artifact.plot_writer import save_figure


def write_rolling_backtest_schematic(ctx: ExperimentPipelineContext) -> None:
	if ctx.windows is None or ctx.windows.empty:
		raise ValueError()

	history_end = _history_end_from_context(ctx)
	holdout_tail_weeks = ctx.manifest.backtest.resolve_holdout_tail_weeks()
	output_path = ctx.artifacts_paths_map.backtest_file("rolling_backtest_schematic.png")
	save_rolling_backtest_schematic(
		windows=ctx.windows,
		output_path=output_path,
		holdout_tail_weeks=holdout_tail_weeks,
		history_end=history_end,
	)


def save_rolling_backtest_schematic(
	windows: pd.DataFrame,
	output_path: Path,
	holdout_tail_weeks: int,
	history_end: pd.Timestamp | None = None,
) -> None:
	if windows.empty:
		raise ValueError()

	windows = _normalize_window_dates(windows)
	sample = _select_schematic_windows(windows)
	if sample.empty:
		return

	if history_end is None:
		history_end = pd.Timestamp(windows["test_start"].max())
	else:
		history_end = pd.Timestamp(history_end)
	history_start = pd.Timestamp(windows["train_start"].min())
	holdout_start = history_end - pd.Timedelta(weeks=int(holdout_tail_weeks))

	fig, (ax_timeline, ax_zoom, ax_holdout) = plt.subplots(
		3,
		1,
		figsize=(15, max(13.0, 0.72 * len(sample) + 7.5)),
		gridspec_kw={"height_ratios": [3.0, 2.2, 1.4]},
	)
	fig.suptitle("Схема rolling backtest windows", fontsize=14, fontweight="bold")

	for idx, row in enumerate(sample.itertuples(index=False), start=1):
		y = idx
		for axis in (ax_timeline, ax_zoom):
			axis.plot([row.train_start, row.train_end], [y, y], color="#1f4e79", linewidth=4)
			axis.plot(
				[row.train_end, row.test_start],
				[y, y],
				color="#f4a261",
				linewidth=2,
				linestyle="--",
			)
			axis.scatter([row.forecast_origin], [y], color="#2a9d8f", s=40, zorder=3)
			axis.scatter([row.test_start], [y], color="#d1495b", s=52, marker="s", zorder=3)

	origin_count = int(windows["forecast_origin"].nunique())
	horizon_count = int(windows["horizon"].nunique())
	window_count = int(len(windows))
	sampled_origin_count = int(sample["forecast_origin"].nunique())
	ax_timeline.set_title(
		"Почему windows много"
	)
	ax_timeline.text(
		0.01,
		0.98,
		(
			f"Всего windows = forecast origins ({origin_count}) x горизонты ({horizon_count}) = {window_count}. "
			f"На графике показан sample: {sampled_origin_count} origin-среза x {horizon_count} горизонтов."
		),
		transform=ax_timeline.transAxes,
		fontsize=9,
		color="#444444",
		va="top",
		bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92, "pad": 5},
	)
	ax_timeline.set_xlabel("week_start")
	ax_timeline.set_ylabel("sampled window")
	ax_timeline.grid(alpha=0.2)
	ax_timeline.legend(
		handles=[
			plt.Line2D([0], [0], color="#1f4e79", linewidth=4, label="train: история до origin"),
			plt.Line2D([0], [0], color="#f4a261", linewidth=2, linestyle="--", label="gap: горизонт до target"),
			plt.Line2D([0], [0], color="#2a9d8f", marker="o", linestyle="None", label="forecast origin"),
			plt.Line2D([0], [0], color="#d1495b", marker="s", linestyle="None", label="target week"),
			plt.Line2D(
				[0],
				[0],
				color="#6c757d",
				linewidth=2,
				linestyle=":",
				label="cutoff holdout",
			),
		],
		loc="lower right",
	)

	zoom_start = sample["forecast_origin"].min() - pd.Timedelta(weeks=16)
	zoom_end = sample["test_start"].max() + pd.Timedelta(weeks=4)
	ax_zoom.set_xlim(zoom_start, zoom_end)
	ax_zoom.set_title(
		"Как делится одно окно"
	)
	ax_zoom.text(
		0.01,
		0.98,
		(
			"В каждом window модель видит только синюю историю до forecast_origin. "
			"Красная target-неделя лежит правее на величину horizon; оранжевый пунктир не входит в train."
		),
		transform=ax_zoom.transAxes,
		fontsize=9,
		color="#444444",
		va="top",
		bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92, "pad": 5},
	)
	ax_zoom.set_xlabel("week_start")
	ax_zoom.set_ylabel("sampled window")
	ax_zoom.grid(alpha=0.2)
	for axis in (ax_timeline, ax_zoom):
		axis.axvline(holdout_start, color="#6c757d", linestyle=":", linewidth=2)

	ax_holdout.plot([history_start, holdout_start], [1, 1], color="#1f4e79", linewidth=6)
	if holdout_tail_weeks > 0:
		ax_holdout.plot(
			[holdout_start, history_end],
			[1, 1],
			color="#d1495b",
			linewidth=4,
			linestyle="--",
		)
	ax_holdout.axvline(holdout_start, color="#6c757d", linestyle=":", linewidth=2)
	ax_holdout.axvline(history_end, color="#343a40", linestyle=":", linewidth=1.5)
	ax_holdout.scatter([holdout_start], [1], color="#6c757d", s=52, zorder=3)
	ax_holdout.scatter([history_end], [1], color="#343a40", s=42, zorder=3)
	ax_holdout.text(
		holdout_start,
		1.08,
		f"history cutoff: {holdout_start.date()}",
		fontsize=9,
		color="#444444",
	)
	ax_holdout.text(
		history_end,
		0.88,
		f"observed end: {history_end.date()}",
		fontsize=9,
		color="#444444",
		ha="right",
	)
	ax_holdout.set_xlim(history_start - pd.Timedelta(weeks=4), history_end + pd.Timedelta(weeks=4))
	ax_holdout.set_ylim(0.75, 1.25)
	ax_holdout.set_yticks([])
	ax_holdout.set_title(
		f"Holdout для финального forecast: последние {int(holdout_tail_weeks)} недель не входят в backtest targets"
	)
	ax_holdout.text(
		0.01,
		0.96,
		(
			"Backtest заканчивает target-недели на cutoff. Красный пунктир справа нужен "
			"для проверки финального прогноза, а не для rolling evaluation."
		),
		transform=ax_holdout.transAxes,
		fontsize=9,
		color="#444444",
		va="top",
		bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.92, "pad": 5},
	)
	ax_holdout.set_xlabel("week_start")
	ax_holdout.grid(axis="x", alpha=0.2)

	save_figure(fig, output_path, dpi=160)


def _normalize_window_dates(windows: pd.DataFrame) -> pd.DataFrame:
	normalized = windows.copy()
	for column in ("forecast_origin", "train_start", "train_end", "test_start", "test_end"):
		normalized[column] = pd.to_datetime(normalized[column])
	return normalized


def _history_end_from_context(ctx: ExperimentPipelineContext) -> pd.Timestamp | None:
	if ctx.dataset is None or ctx.dataset.weekly.empty:
		return None
	if "week_start" not in ctx.dataset.weekly.columns:
		return None
	return pd.Timestamp(ctx.dataset.weekly["week_start"].max())


def _select_schematic_windows(windows: pd.DataFrame) -> pd.DataFrame:
	ordered = windows.sort_values(["forecast_origin", "horizon"]).copy()
	origins = ordered["forecast_origin"].drop_duplicates().tolist()
	if not origins:
		return ordered.head(0)

	sampled_origins = []
	for position in (0, len(origins) // 2, len(origins) - 1):
		origin = origins[position]
		if origin not in sampled_origins:
			sampled_origins.append(origin)

	return (
		ordered[ordered["forecast_origin"].isin(sampled_origins)]
		.sort_values(["forecast_origin", "horizon"])
		.reset_index(drop=True)
	)
