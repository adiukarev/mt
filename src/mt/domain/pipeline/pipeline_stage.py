from abc import ABC, abstractmethod
import re
import time

from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.infra.helper.str import to_snake_case_without_suffix
from mt.infra.observability.runtime.bootstrap import attach_observability
from mt.infra.observability.runtime.stage_events import log_stage_start, log_stage_end


class BasePipelineStage(ABC):
	@property
	def name(self) -> str:
		return to_snake_case_without_suffix(self.__class__.__name__, "PipelineStage")

	def run(self, ctx: BasePipelineContext) -> None:
		attach_observability(ctx)

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
