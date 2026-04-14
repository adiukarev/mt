from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mt.domain.dataset import DatasetBundle, DatasetLoadData
from mt.domain.manifest import DatasetManifest
from mt.domain.pipeline_context import BasePipelineContext


@dataclass(slots=True)
class AuditPipelineContext(BasePipelineContext):
	"""Контекст пайплайна аудита данных"""

	dataset_manifest: DatasetManifest
	output_dir: Path
	raw_dataset: DatasetLoadData | None = None
	dataset: DatasetBundle | None = None
	segments: pd.DataFrame | None = None
	raw_context: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AuditArtifactEntry:
	path: str
	title: str
	kind: str
	purpose: str
	scope_note: str


@dataclass(slots=True)
class AuditArtifacts:
	summary: pd.DataFrame
	dataset_profile: pd.DataFrame
	aggregation_comparison: pd.DataFrame
	segment_summary: pd.DataFrame
	category_summary: pd.DataFrame
	sku_summary: pd.DataFrame
	category_correlation_matrix: pd.DataFrame
	category_growth_correlation_matrix: pd.DataFrame
	category_seasonal_index: pd.DataFrame
	sku_concentration_summary: pd.DataFrame
	sku_share_stability_summary: pd.DataFrame
	feature_availability: pd.DataFrame
	feature_block_summary: pd.DataFrame
	seasonality_summary: pd.DataFrame
	diagnostic_summary: pd.DataFrame
	stationarity_summary: pd.DataFrame
	data_dictionary: pd.DataFrame
	transformation_summary: pd.DataFrame
	example_feature_snapshots: dict[str, pd.DataFrame]
	report_lines: list[str]
