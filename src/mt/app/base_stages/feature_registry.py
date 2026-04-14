from abc import ABC

from mt.domain.stage import BaseStage
from mt.infra.feature.registry import build_feature_registry


class BaseFeatureRegistryStage(BaseStage, ABC):
	"""Общий шаг построения реестра признаков"""

	def execute(self, ctx: object) -> None:
		dataset = getattr(ctx, "dataset", None)
		if dataset is None:
			raise ValueError()

		manifest = self.get_feature_manifest(ctx)
		ctx.feature_registry = build_feature_registry(manifest, dataset.aggregation_level)

		self.after_execute(ctx)

	def get_feature_manifest(self, ctx: object):
		return ctx.feature_manifest

	def after_execute(self, ctx: object) -> None:
		return None
