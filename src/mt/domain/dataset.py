from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mt.domain.manifest import DatasetManifest


@dataclass(slots=True)
class DatasetLoadData:
	"""Исходные таблицы датасета M5"""

	sales: pd.DataFrame
	calendar: pd.DataFrame
	sell_prices: pd.DataFrame


@dataclass(slots=True)
class DatasetBundle:
	"""Подготовленные таблицы временных рядов и метаданные"""

	# Уровень агрегации ряда
	aggregation_level: str
	# Название целевой переменной
	target_name: str
	# Недели
	weekly: pd.DataFrame
	# Дополнительные метаданные датасета
	metadata: dict[str, Any] = field(default_factory=dict)


class DatasetBundler[TData](ABC):
	"""Базовый адаптер загрузки и подготовки датасета"""

	def __init__(self, manifest: DatasetManifest):
		"""Сохранить настройки датасета"""

		self.manifest = manifest

	@abstractmethod
	def load(self) -> TData:
		"""Загрузить исходные таблицы датасета"""

		...

	@abstractmethod
	def prepare(self, data: TData) -> DatasetBundle:
		"""Преобразовать исходные таблицы в стандартный bundle"""

		...
