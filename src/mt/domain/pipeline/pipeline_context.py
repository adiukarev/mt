from dataclasses import dataclass, field
from typing import TypeVar

from mt.domain.observability.observability import ObservabilityContext

T = TypeVar("T")


@dataclass(slots=True)
class BasePipelineContext:
	"""Общий стейт контекст между этапами"""

	# Выполненные этапы в хронологическом порядке
	executed_stages: list[str] = field(default_factory=list, init=False)
	# Время выполнения этапов пайплайна
	stage_timings: list[dict[str, object]] = field(default_factory=list, init=False)
	# Полное время выполнения пайплайна
	pipeline_wall_time_seconds: float | None = field(default=None, init=False)
	# Легковесные служебные метаданные orchestration/tracking слоя
	runtime_metadata: dict[str, object] = field(default_factory=dict, init=False)
	# Typed observability context shared by logging, tracking and artifacts
	observability: ObservabilityContext | None = field(default=None, init=False)

	def require_value(self, value: T | None, field_name: str) -> T:
		if value is None:
			raise ValueError(f"{field_name} must be available")
		return value
