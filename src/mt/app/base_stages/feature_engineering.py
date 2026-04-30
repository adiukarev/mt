from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.feature.registry_builder import build_feature_registry
from mt.infra.validation.feature import validate_feature_registry


class BaseFeatureEngineeringPipelineStage(BasePipelineStage):
	def execute(self, ctx: object) -> None:
		ctx.feature_registry = build_feature_registry(
			ctx.feature_manifest,
			ctx.dataset.aggregation_level
		)

		validate_feature_registry(ctx.feature_registry)
