from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.runtime.runtime_manifest import RuntimeManifest
from mt.infra.artifact.text_reader import read_yaml_mapping
from mt.infra.helper.dict import get_required_mapping_section


@dataclass(slots=True)
class AuditPipelineManifest:
	dataset: DatasetManifest = field(default_factory=DatasetManifest)
	runtime: RuntimeManifest = field(default_factory=RuntimeManifest)

	def __post_init__(self) -> None:
		if not self.runtime.artifacts_dir:
			self.runtime.artifacts_dir = f"artifacts/audit_{self.dataset.aggregation_level}"

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)

	@staticmethod
	def from_dict(data: dict[str, Any]) -> "AuditPipelineManifest":
		manifest = AuditPipelineManifest(
			dataset=DatasetManifest(**get_required_mapping_section(data, "dataset")),
			runtime=RuntimeManifest(**get_required_mapping_section(data, "runtime")),
		)
		return manifest

	@staticmethod
	def load(source: str | Path | dict[str, Any]) -> "AuditPipelineManifest":
		if isinstance(source, dict):
			return AuditPipelineManifest.from_dict(source)

		return AuditPipelineManifest.from_dict(read_yaml_mapping(source))

