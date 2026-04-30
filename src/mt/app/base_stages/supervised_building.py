from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.feature.supervised_builder import build_supervised_frame


class BaseSupervisedBuildingPipelineStage(BasePipelineStage):
	"""Шаг где собираем supervised и target_h* (таблица с признаками, конечная перед моделью)"""

	def execute(self, ctx: object) -> None:
		ctx.supervised, ctx.feature_columns = build_supervised_frame(
			ctx.dataset.weekly,
			ctx.segments,
			ctx.feature_manifest,
			ctx.backtest_manifest,
		)
