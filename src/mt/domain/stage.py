from abc import ABC, abstractmethod
import time

from mt.domain.pipeline_context import BasePipelineContext
from mt.infra.artifact.logs.stage import log_stage_start, log_stage_end


class BaseStage(ABC):
	name: str

	def run(self, ctx: BasePipelineContext) -> None:
		log_stage_start(self.name)
		stage_started_at = time.perf_counter()

		self.execute(ctx)
		wall_time_seconds = time.perf_counter() - stage_started_at

		ctx.executed_stages.append(self.name)
		ctx.stage_timings.append(
			{
				"stage_name": self.name,
				"status": "completed",
				"wall_time_seconds": wall_time_seconds,
			}
		)

		log_stage_end(self.name, wall_time_seconds)

	@abstractmethod
	def execute(self, ctx: BasePipelineContext) -> None:
		...
