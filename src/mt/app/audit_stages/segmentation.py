from mt.domain.stage import BaseStage
from mt.infra.feature.segmentation import segment_series

from mt.domain.audit import AuditPipelineContext


class AuditSegmentationStage(BaseStage):
	name = "audit_segmentation"

	def execute(self, ctx: AuditPipelineContext) -> None:
		if ctx.dataset is None:
			raise ValueError()

		# для описания неоднородности рядов и последующего анализа качества по типам серий
		ctx.segments = segment_series(ctx.dataset.weekly)
