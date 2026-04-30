from dataclasses import asdict

import pandas as pd

from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.backtest.windows_builder import build_backtest_windows


class ExperimentBacktestWindowPreparationPipelineStage(BasePipelineStage):
	def execute(self, ctx: ExperimentPipelineContext) -> None:
		if ctx.dataset is None:
			raise ValueError()

		# тут лучше упасть сразу, если лаги/горизонт уже не лезут в train
		self._validate_backtest_readiness(ctx)

		# окна общие для всех моделей
		# feature_set пишем просто как след от общего набора, не как реальный список колонок модели
		ctx.windows = pd.DataFrame(
			asdict(window) for window in
			build_backtest_windows(ctx.manifest.backtest, ctx.dataset.weekly)
		)
		if ctx.windows is not None and not ctx.windows.empty:
			ctx.windows["feature_set"] = ctx.feature_manifest.feature_set

		if ctx.windows is None or ctx.windows.empty \
			or not bool((ctx.windows["train_end"] < ctx.windows["test_start"]).all()) \
			or not bool((ctx.windows["test_start"] == ctx.windows["test_end"]).all()) \
			or not bool(
			ctx.windows["feature_set"].nunique() == 1
			and ctx.windows["feature_set"].iloc[0] == ctx.feature_manifest.feature_set
		):
			raise ValueError()

	def _validate_backtest_readiness(self, ctx: ExperimentPipelineContext) -> None:
		max_lag = max(ctx.feature_manifest.lags, default=0) if ctx.feature_manifest.enabled else 0
		min_required_train_weeks = max_lag + ctx.manifest.backtest.horizon_end + 1

		if ctx.manifest.backtest.min_train_weeks < min_required_train_weeks:
			raise ValueError()
