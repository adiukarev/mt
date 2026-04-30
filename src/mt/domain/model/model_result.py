from dataclasses import dataclass, field

import pandas as pd

from mt.domain.model.model_info import ModelInfo


@dataclass(slots=True)
class ModelResult:
	"""Результат выполнения модели на backtest/inference контуре"""

	info: ModelInfo
	predictions: pd.DataFrame
	warnings: list[str] = field(default_factory=list)
	train_time_seconds: float | None = None
	inference_time_seconds: float | None = None
	wall_time_seconds: float | None = None
	used_feature_columns: list[str] = field(default_factory=list)
	calibration_summary: pd.DataFrame | None = None
	probabilistic_metadata: dict[str, object] = field(default_factory=dict)
