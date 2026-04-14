from abc import ABC, abstractmethod

from mt.domain.manifest import DatasetManifest
from mt.domain.stage import BaseStage
from mt.infra.dataset.prepare import prepare_dataset


class BaseDatasetPreparationStage(BaseStage, ABC):
	"""Общий шаг превращения raw-таблиц в недельный dataset"""

	def execute(self, ctx: object) -> None:
		raw_dataset = getattr(ctx, "raw_dataset", None)
		if raw_dataset is None:
			raise ValueError()

		ctx.dataset = prepare_dataset(self.get_dataset_manifest(ctx), raw_dataset)

		self.after_execute(ctx)

	@abstractmethod
	def get_dataset_manifest(self, ctx: object) -> DatasetManifest:
		...

	def after_execute(self, ctx: object) -> None:
		return None
