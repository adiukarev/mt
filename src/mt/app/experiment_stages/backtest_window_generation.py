from dataclasses import asdict

import pandas as pd

from mt.domain.experiment import ExperimentPipelineContext
from mt.domain.stage import BaseStage
from mt.infra.artifact.writer import write_csv, write_markdown
from mt.infra.artifact.experiment_paths import experiment_artifact_relpath
from mt.infra.backtest.backtest import build_backtest_windows


class BacktestWindowGenerationStage(BaseStage):
	name = "experiment_backtest_window_generation"

	def execute(self, ctx: ExperimentPipelineContext) -> None:
		if ctx.dataset is None:
			raise ValueError()

		# тут лучше упасть сразу, если лаги/горизонт уже не лезут в train
		self._validate_backtest_readiness(ctx)

		# окна общие для всех моделей
		# feature_set пишем просто как след от общего набора, не как реальный список колонок модели
		windows = build_backtest_windows(
			ctx.dataset.weekly,
			ctx.dataset.aggregation_level,
			ctx.manifest.backtest,
			ctx.feature_manifest.feature_set,
			ctx.manifest.runtime.seed,
		)
		ctx.windows = pd.DataFrame(asdict(window) for window in windows)

		if ctx.windows is None or ctx.windows.empty \
			or not bool((ctx.windows["train_end"] < ctx.windows["test_start"]).all()) \
			or not bool((ctx.windows["test_start"] == ctx.windows["test_end"]).all()) \
			or not bool(
			ctx.windows["feature_set"].nunique() == 1
			and ctx.windows["feature_set"].iloc[0] == ctx.feature_manifest.feature_set
		):
			raise ValueError()

		self._persist_artifacts(ctx)

	def _persist_artifacts(self, ctx: ExperimentPipelineContext) -> None:
		write_csv(
			ctx.artifacts_paths_map.root / experiment_artifact_relpath("backtest_windows.csv"),
			ctx.windows
		)
		write_markdown(
			ctx.artifacts_paths_map.root / experiment_artifact_relpath("backtest_window_generation.md"),
			[
				"# Генерация окон backtest",
				"",
				f"- количество окон: {len(ctx.windows)}",
				f"- горизонты: {sorted(ctx.windows['horizon'].unique().tolist())}",
				f"- provenance общего supervised superset: {ctx.feature_manifest.feature_set}",
				"- окна одинаковы для всех моделей; различия по признакам фиксируются в model manifests и model_feature_usage.csv",
			],
		)

	def _validate_backtest_readiness(self, ctx: ExperimentPipelineContext) -> None:
		max_lag = max(ctx.feature_manifest.lags, default=0) if ctx.feature_manifest.enabled else 0
		min_required_train_weeks = max_lag + ctx.manifest.backtest.horizon_max + 1

		if ctx.manifest.backtest.min_train_weeks < min_required_train_weeks:
			raise ValueError()
