from abc import ABC, abstractmethod
import time

from mt.domain.pipeline_context import BasePipelineContext
from mt.domain.stage import BaseStage


class BasePipeline(ABC):
	"""Базовый пайплайн последовательного выполнения этапов"""

	def __init__(self, stages: list[BaseStage]):
		self.stages = stages

	def run(self, *args, **kwargs) -> BasePipelineContext:
		"""Построить контекст, выполнить этапы и вернуть финальное состояние"""

		ctx = self.build_context(*args, **kwargs)
		pipeline_started_at = time.perf_counter()

		for stage in self.stages:
			stage.run(ctx)

		ctx.pipeline_wall_time_seconds = time.perf_counter() - pipeline_started_at

		return self.finalize(ctx)

	@abstractmethod
	def build_context(self, *args, **kwargs) -> BasePipelineContext:
		"""Создать начальный контекст пайплайна"""

		...

	@abstractmethod
	def finalize(self, ctx: BasePipelineContext) -> BasePipelineContext:
		"""Сохранить итоговые артефакты и завершить пайплайн"""

		...
