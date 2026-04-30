import pandas as pd

from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.infra.artifact.text_writer import write_csv, write_markdown


def write_backtest_windows(ctx: ExperimentPipelineContext) -> None:
	write_csv(ctx.artifacts_paths_map.backtest_file("backtest_windows.csv"), _enriched_windows(ctx))


def write_backtest_windows_by_horizon(ctx: ExperimentPipelineContext) -> None:
	windows = _enriched_windows(ctx)
	windows_by_horizon = (
		windows.groupby("horizon", as_index=False)
		.agg(
			window_count=("horizon", "count"),
			forecast_origin_count=("forecast_origin", "nunique"),
			first_forecast_origin=("forecast_origin", "min"),
			last_forecast_origin=("forecast_origin", "max"),
			first_target_week=("target_week", "min"),
			last_target_week=("target_week", "max"),
			min_train_weeks=("train_weeks", "min"),
			max_train_weeks=("train_weeks", "max"),
			gap_weeks=("gap_weeks", "max"),
		)
	)
	write_csv(
		ctx.artifacts_paths_map.backtest_file("backtest_windows_by_horizon.csv"),
		windows_by_horizon,
	)


def write_backtest_window_summary(ctx: ExperimentPipelineContext) -> None:
	if ctx.dataset is None or ctx.windows is None or ctx.windows.empty:
		raise ValueError()

	windows = _enriched_windows(ctx)
	history_end = pd.Timestamp(ctx.dataset.weekly["week_start"].max())
	holdout_tail_weeks = ctx.manifest.backtest.resolve_holdout_tail_weeks()
	holdout_start = history_end - pd.Timedelta(weeks=holdout_tail_weeks)
	window_summary = pd.DataFrame(
		[
			{
				"aggregation_level": ctx.dataset.aggregation_level,
				"feature_superset": ctx.feature_manifest.feature_set,
				"horizon_start": ctx.manifest.backtest.horizon_start,
				"horizon_end": ctx.manifest.backtest.horizon_end,
				"window_count": len(windows),
				"origin_count": windows["forecast_origin"].nunique(),
				"shared_origin_grid": ctx.manifest.backtest.shared_origin_grid,
				"origin_step_weeks": ctx.manifest.backtest.step_weeks,
				"min_train_weeks": int(windows["train_weeks"].min()),
				"max_train_weeks": int(windows["train_weeks"].max()),
				"forecast_origin_start": windows["forecast_origin"].min(),
				"forecast_origin_end": windows["forecast_origin"].max(),
				"target_week_start": windows["target_week"].min(),
				"target_week_end": windows["target_week"].max(),
				"history_observed_end": history_end,
				"holdout_tail_weeks": holdout_tail_weeks,
				"history_cutoff_for_final_training": holdout_start,
				"backtest_targets_exclude_holdout_tail": holdout_tail_weeks > 0,
				"availability_rule": "all features must be known at forecast origin",
			}
		]
	)
	write_csv(
		ctx.artifacts_paths_map.backtest_file("backtest_window_summary.csv"),
		window_summary,
	)


def write_backtest_window_train_test_counts(ctx: ExperimentPipelineContext) -> None:
	windows = _enriched_windows(ctx)
	write_csv(
		ctx.artifacts_paths_map.backtest_file("backtest_window_train_test_counts.csv"),
		windows.loc[
			:,
			[
				"horizon",
				"forecast_origin",
				"train_end",
				"target_week",
				"gap_weeks",
				"train_weeks",
				"test_weeks",
			],
		].copy(),
	)


def write_backtest_window_generation_summary(ctx: ExperimentPipelineContext) -> None:
	windows = _enriched_windows(ctx)
	first_window = windows.sort_values(["forecast_origin", "horizon"]).iloc[0]
	history_end = pd.Timestamp(ctx.require_dataset().weekly["week_start"].max())
	holdout_tail_weeks = ctx.manifest.backtest.resolve_holdout_tail_weeks()
	holdout_start = history_end - pd.Timedelta(weeks=holdout_tail_weeks)
	write_markdown(
		ctx.artifacts_paths_map.backtest_file("backtest_window_generation.md"),
		[
			"# Генерация окон backtest",
			"",
			f"- Количество окон: {len(windows)}",
			f"- Горизонты: {sorted(windows['horizon'].unique().tolist())}",
			f"- Уникальных forecast origins: {windows['forecast_origin'].nunique()}",
			f"- Общая origin-grid для всех горизонтов: {'yes' if ctx.manifest.backtest.shared_origin_grid else 'no'}",
			f"- Минимальный размер train, недель: {int(windows['train_weeks'].min())}",
			f"- Максимальный размер train, недель: {int(windows['train_weeks'].max())}",
			f"- Шаг между origins, недель: {ctx.manifest.backtest.step_weeks}",
			f"- Конец наблюдаемой истории: {history_end.date()}",
			f"- Holdout-tail для финального обучения, недель: {holdout_tail_weeks}",
			f"- Cutoff истории для финального обучения: {holdout_start.date()}",
			"- Backtest target weeks не заходят в holdout-tail; последний target должен быть не правее cutoff.",
			f"- provenance общего supervised superset: {ctx.feature_manifest.feature_set}",
			"- Окна одинаковы для всех моделей; различия по признакам фиксируются в model manifests и preparation/08_model_feature_usage.csv",
			"- Для каждого горизонта `forecast_origin` является последней доступной исторической неделей окна, а тест относится к неделе `forecast_origin + horizon`.",
			"- History-based/statistical/baseline модели могут использовать наблюденные `sales_units` до `forecast_origin`; direct supervised ML-модели обучаются только на строках, где `week_start + horizon <= forecast_origin`, чтобы `target_h*` не включал будущий факт оцениваемого окна.",
			"- Нижняя панель `rolling_backtest_schematic.png` показывает пунктиром хвост истории, который не используется как доступная история при финальном forecast-origin.",
			"- Это гарантирует, что target для train не использует будущие недели, которые позже станут объектом оценки в том же окне.",
			"- train/test разделяются по целевым неделям и не пересекаются.",
			"- Лаги и rolling-статистики строятся только по прошлым наблюдениям относительно forecast origin.",
			"- Общая origin-grid делает метрики по горизонтам сопоставимыми: каждый горизонт оценивается на одном и том же наборе forecast origins.",
			"- rolling window backtesting предпочтительнее одиночного split, потому что проверяет устойчивость качества на множестве последовательных forecast origins.",
			"",
			"## Пример одного окна",
			f"- forecast_origin: {first_window['forecast_origin'].date()}",
			f"- horizon: {int(first_window['horizon'])}",
			f"- train_span: {first_window['train_start'].date()} .. {first_window['train_end'].date()}",
			f"- target_week: {first_window['target_week'].date()}",
			f"- gap_weeks_between_train_end_and_target: {int(first_window['gap_weeks'])}",
		],
	)


def _enriched_windows(ctx: ExperimentPipelineContext) -> pd.DataFrame:
	if ctx.windows is None or ctx.windows.empty:
		raise ValueError()

	windows = ctx.windows.copy()
	windows["target_week"] = windows["test_start"]
	windows["train_weeks"] = ((windows["train_end"] - windows["train_start"]).dt.days // 7) + 1
	windows["test_weeks"] = ((windows["test_end"] - windows["test_start"]).dt.days // 7) + 1
	windows["gap_weeks"] = ((windows["target_week"] - windows["train_end"]).dt.days // 7)
	return windows
