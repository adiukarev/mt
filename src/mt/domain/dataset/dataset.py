from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mt.domain.dataset.dataset_kind import DatasetKind
from mt.domain.dataset.dataset_manifest import DatasetManifest


@dataclass(slots=True)
class DatasetLoadData:
	"""Dataset-specific raw tables loaded from disk."""

	kind: DatasetKind
	tables: dict[str, pd.DataFrame]
	metadata: dict[str, Any] = field(default_factory=dict)

	def require_table(self, name: str) -> pd.DataFrame:
		if name not in self.tables:
			raise KeyError()
		return self.tables[name]


@dataclass(slots=True)
class DatasetBundle:
	"""Canonical weekly dataset for downstream pipelines."""

	kind: DatasetKind
	adapter_name: str
	aggregation_level: str
	target_name: str
	weekly: pd.DataFrame
	metadata: dict[str, Any] = field(default_factory=dict)
	policy_flags: dict[str, Any] = field(default_factory=dict)
	raw_context: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
	"""Dataset adapter for raw ingestion, normalization and forecast framing."""

	def __init__(self, manifest: DatasetManifest):
		self.manifest = manifest

	@property
	@abstractmethod
	def kind(self) -> DatasetKind:
		...

	@abstractmethod
	def load(self) -> DatasetLoadData:
		...

	@abstractmethod
	def prepare(self, data: DatasetLoadData) -> DatasetBundle:
		...

	@abstractmethod
	def build_raw_context(
		self,
		data: DatasetLoadData,
		bundle: DatasetBundle,
	) -> dict[str, Any]:
		...
