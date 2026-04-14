import logging


def log_stage_start(stage_name: str) -> None:
	logging.info(f"-> {stage_name} started")


def log_stage_end(stage_name: str, wall_time_seconds: float) -> None:
	logging.info(f"<- {stage_name} completed | wall_time={wall_time_seconds:.3f}s")
