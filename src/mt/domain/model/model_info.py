from dataclasses import dataclass

from mt.domain.model.model_family import ModelFamily
from mt.domain.model.model_name import ModelName


@dataclass(slots=True)
class ModelInfo:
	"""Сопоставимые метаданные модели"""

	model_name: ModelName
	model_family: ModelFamily
