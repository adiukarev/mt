from dataclasses import dataclass, field


@dataclass(slots=True)
class BasePipelineContext:
	"""Общий стейт контекст между этапами"""

	# Выполненные этапы в хронологическом порядке
	executed_stages: list[str] = field(default_factory=list, init=False)
	# Время выполнения этапов пайплайна
	stage_timings: list[dict[str, object]] = field(default_factory=list, init=False)
	# Полное время выполнения пайплайна
	pipeline_wall_time_seconds: float | None = field(default=None, init=False)
