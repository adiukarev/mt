from abc import ABC

from mt.domain.stage import BaseStage
from mt.infra.feature.segmentation import segment_series


class BaseSegmentationStage(BaseStage, ABC):
	"""Общий шаг сегментации временных рядов"""

	def execute(self, ctx: object) -> None:
		dataset = getattr(ctx, "dataset", None)
		if dataset is None:
			raise ValueError()

		ctx.segments = segment_series(dataset.weekly)

		self.after_execute(ctx)

	def after_execute(self, ctx: object) -> None:
		return None
