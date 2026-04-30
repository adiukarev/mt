from dataclasses import dataclass, field
import pandas as pd

from mt.domain.audit.audit_pipeline_manifest import AuditPipelineManifest
from mt.domain.audit.audit_artifact import AuditArtifactData
from mt.domain.dataset.dataset import DatasetBundle, DatasetLoadData
from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.domain.audit.audit_artifact import AuditArtifactPathsMap


@dataclass(slots=True)
class AuditPipelineContext(BasePipelineContext):
	manifest: AuditPipelineManifest

	artifacts_paths_map: AuditArtifactPathsMap | None = None

	dataset: DatasetBundle | None = None
	raw_dataset: DatasetLoadData | None = None

	raw_context: dict[str, object] = field(default_factory=dict)

	segments: pd.DataFrame | None = None

	audit_artifacts: AuditArtifactData | None = None

	def __post_init__(self) -> None:
		self.artifacts_paths_map = AuditArtifactPathsMap.ensure(self.manifest.runtime.artifacts_dir)

	@property
	def dataset_manifest(self) -> object:
		return self.manifest.dataset

	def require_dataset(self) -> DatasetBundle:
		return self.require_value(self.dataset, "dataset")

	def require_raw_dataset(self) -> DatasetLoadData:
		return self.require_value(self.raw_dataset, "raw_dataset")

	def require_segments(self) -> pd.DataFrame:
		return self.require_value(self.segments, "segments")

	def require_audit_artifacts(self) -> AuditArtifactData:
		return self.require_value(self.audit_artifacts, "audit_artifacts")
