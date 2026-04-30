from dataclasses import dataclass, field
from typing import Any

from mt.domain.dataset.dataset_source_type import (
	ALLOWED_DATASET_SOURCE_TYPES,
	DatasetSourceType,
	normalize_dataset_source_type,
)


@dataclass(slots=True)
class DatasetSourceManifest:
	source_type: DatasetSourceType = DatasetSourceType.LOCAL
	source_config: dict[str, Any] = field(default_factory=dict)

	def __post_init__(self) -> None:
		self.source_type = normalize_dataset_source_type(self.source_type)
		if self.source_type not in ALLOWED_DATASET_SOURCE_TYPES:
			raise ValueError()
		if not isinstance(self.source_config, dict):
			raise TypeError("source_config must be a mapping")
