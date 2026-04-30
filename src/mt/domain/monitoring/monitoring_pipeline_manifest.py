from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.forecast.forecast_pipeline_manifest import (
	ForecastLocalModelManifest,
	ForecastModelManifest,
	ForecastRegistryModelManifest,
)
from mt.domain.model.registry_selection_manifest import RegistrySelectionManifest
from mt.domain.runtime.runtime_manifest import RuntimeManifest
from mt.infra.artifact.text_reader import read_yaml_mapping
from mt.infra.helper.dict import get_required_mapping_section


@dataclass(slots=True)
class MonitoringDriftManifest:
	reference_weeks: int = 4
	max_distribution_shift_score: float = 0.30
	max_zero_share_delta: float = 0.12
	max_row_count_delta: float = 0.20

	def __post_init__(self) -> None:
		if self.reference_weeks < 1:
			raise ValueError()


@dataclass(slots=True)
class MonitoringQualityGateManifest:
	max_recent_wape: float = 0.35
	max_alert_score: float = 2.0

	def __post_init__(self) -> None:
		if self.max_recent_wape <= 0:
			raise ValueError()
		if self.max_alert_score <= 0:
			raise ValueError()


@dataclass(slots=True)
class MonitoringPipelineManifest:
	dataset: DatasetManifest
	model: ForecastModelManifest
	drift: MonitoringDriftManifest = field(default_factory=MonitoringDriftManifest)
	quality_gate: MonitoringQualityGateManifest = field(
		default_factory=MonitoringQualityGateManifest
	)
	runtime: RuntimeManifest = field(default_factory=RuntimeManifest)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)

	@staticmethod
	def from_dict(data: dict[str, Any]) -> "MonitoringPipelineManifest":
		dataset_payload = data.get("dataset")
		if not isinstance(dataset_payload, dict):
			raise ValueError("Monitoring manifest requires top-level dataset section")
		model_payload = get_required_mapping_section(data, "model")
		registry_payload = dict(model_payload.get("registry", {}))
		selection_payload = dict(registry_payload.get("selection", {}))
		selection_alias = selection_payload.get("alias", selection_payload.get("registry_alias", "dev"))
		return MonitoringPipelineManifest(
			dataset=DatasetManifest(**dataset_payload),
			model=ForecastModelManifest(
				source_preference=model_payload.get("source_preference", "auto"),
				local=ForecastLocalModelManifest(**dict(model_payload.get("local", {}))),
				registry=ForecastRegistryModelManifest(
					selection=RegistrySelectionManifest(
						dag_id=selection_payload.get("dag_id"),
						alias=selection_alias,
						metric_name=selection_payload.get("metric_name", "WAPE"),
						higher_is_better=bool(selection_payload.get("higher_is_better", False)),
					),
				),
			),
			drift=MonitoringDriftManifest(**dict(data.get("drift", {}))),
			quality_gate=MonitoringQualityGateManifest(
				**dict(data.get("quality_gate", {}))
			),
			runtime=RuntimeManifest(**get_required_mapping_section(data, "runtime")),
		)

	@staticmethod
	def load(source: str | Path | dict[str, Any]) -> "MonitoringPipelineManifest":
		if isinstance(source, dict):
			return MonitoringPipelineManifest.from_dict(source)
		return MonitoringPipelineManifest.from_dict(read_yaml_mapping(source))
