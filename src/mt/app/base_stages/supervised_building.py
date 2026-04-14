from abc import ABC

from mt.domain.stage import BaseStage
from mt.infra.feature.supervised_builder import make_supervised_frame


class BaseSupervisedBuildingStage(BaseStage, ABC):
	"""Шаг где собираем supervised и target_h* (таблица с признаками, конечная перед моделью)"""

	def execute(self, ctx: object) -> None:
		dataset = getattr(ctx, "dataset", None)
		segments = getattr(ctx, "segments", None)
		if dataset is None or segments is None:
			raise ValueError()

		manifest = self.get_feature_manifest(ctx)
		backtest_manifest = self.get_backtest_manifest(ctx)

		ctx.supervised, ctx.feature_columns = make_supervised_frame(
			dataset.weekly,
			segments,
			manifest,
		)

		for horizon in range(backtest_manifest.horizon_min, backtest_manifest.horizon_max + 1):
			ctx.supervised[f"target_h{horizon}"] = ctx.supervised.groupby("series_id")[
				"sales_units"
			].shift(-horizon)

		self.after_execute(ctx)

	def get_feature_manifest(self, ctx: object):
		return ctx.feature_manifest

	def get_backtest_manifest(self, ctx: object):
		return ctx.manifest.backtest

	def after_execute(self, ctx: object) -> None:
		return None
