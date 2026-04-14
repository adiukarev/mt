from abc import ABC, abstractmethod

from mt.domain.manifest import DatasetManifest
from mt.domain.stage import BaseStage
from mt.infra.dataset.load import load_dataset


class BaseDatasetBundlingStage(BaseStage, ABC):
	"""Общий шаг загрузки raw-таблиц датасета"""

	def execute(self, ctx: object) -> None:
		ctx.raw_dataset = load_dataset(self.get_dataset_manifest(ctx))

	@abstractmethod
	def get_dataset_manifest(self, ctx: object) -> DatasetManifest:
		...
