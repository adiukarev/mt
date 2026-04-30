from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.dataset.dataset_source_manifest import DatasetSourceManifest


@dataclass(slots=True)
class DatasetSourceRefreshResult:
	reference_frame: pd.DataFrame
	recent_actuals: pd.DataFrame
	full_frame: pd.DataFrame
	source_descriptor: dict[str, Any] = field(default_factory=dict)
	materialized_paths: list[str] = field(default_factory=list)


class DatasetSourceService(ABC):
	def __init__(self, source_manifest: DatasetSourceManifest):
		self.source_manifest = source_manifest

	@property
	def name(self) -> str:
		return self.__class__.__name__

	@abstractmethod
	def refresh_dataset(
		self,
		dataset_manifest: DatasetManifest,
	) -> DatasetSourceRefreshResult:
		...
