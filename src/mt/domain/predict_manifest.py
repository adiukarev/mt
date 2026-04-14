from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PredictInputManifest:
	"""Входные данные predict pipeline"""

	dataset_path: str
	scenario_name: str | None = None

	def __post_init__(self) -> None:
		if not self.dataset_path:
			raise ValueError()


@dataclass(slots=True)
class PredictModelSourceManifest:
	"""Источник лучшей модели из experiment artifacts"""

	best_model_dir: str

	def __post_init__(self) -> None:
		if not self.best_model_dir:
			raise ValueError()


@dataclass(slots=True)
class PredictForecastManifest:
	"""Параметры горизонта прогноза"""

	horizon_weeks: int | None = None

	def __post_init__(self) -> None:
		if self.horizon_weeks is not None and (self.horizon_weeks < 1 or self.horizon_weeks > 8):
			raise ValueError()


@dataclass(slots=True)
class PredictVisualizationManifest:
	"""Параметры визуализации прогноза"""

	overlay_series_id: str | None = None
	plot_history_weeks: int = 52
	zoom_history_weeks: int = 16
	annotate_forecast_values: bool = True

	def __post_init__(self) -> None:
		if self.plot_history_weeks < 8 or self.zoom_history_weeks < 4 or self.zoom_history_weeks > self.plot_history_weeks:
			raise ValueError()


@dataclass(slots=True)
class PredictRuntimeManifest:
	"""Параметры сохранения predict artifacts"""

	output_dir: str = "artifacts/predict_from_best_model_artifacts"
	seed: int = 42

	def __post_init__(self) -> None:
		if not self.output_dir:
			raise ValueError()


@dataclass(slots=True)
class SyntheticPredictManifest:
	"""Корневой манифест predict pipeline"""

	input: PredictInputManifest
	model_source: PredictModelSourceManifest
	forecast: PredictForecastManifest = field(default_factory=PredictForecastManifest)
	visualization: PredictVisualizationManifest = field(default_factory=PredictVisualizationManifest)
	runtime: PredictRuntimeManifest = field(default_factory=PredictRuntimeManifest)

	def as_dict(self) -> dict[str, Any]:
		return asdict(self)


def load_predict_manifest(path: str | Path) -> SyntheticPredictManifest:
	"""Загрузить YAML-манифест predict pipeline"""

	data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
	return SyntheticPredictManifest(
		input=PredictInputManifest(**_section(data, "input")),
		model_source=PredictModelSourceManifest(**_section(data, "model_source")),
		forecast=PredictForecastManifest(**_section(data, "forecast")),
		visualization=PredictVisualizationManifest(**_section(data, "visualization")),
		runtime=PredictRuntimeManifest(**_section(data, "runtime")),
	)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
	value = data.get(key, {})
	if not isinstance(value, dict):
		raise TypeError()
	return value
