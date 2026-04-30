from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any

from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.model.registry_selection_manifest import (
	RegistrySelectionManifest,
)
from mt.domain.runtime.runtime_manifest import RuntimeManifest
from mt.infra.artifact.text_reader import read_yaml_mapping
from mt.infra.helper.dict import get_required_mapping_section


def _resolve_execution_mode(execution_mode: str | None = None) -> str:
	value = execution_mode or os.getenv("MT_EXECUTION_MODE", "local")
	return value.strip().lower() or "local"


def _normalize_source_preference(value: str | None) -> str:
	resolved = (value or "auto").strip().lower()
	if resolved not in {"auto", "local", "registry"}:
		raise ValueError("source_preference must be one of: auto, local, registry")
	return resolved


@dataclass(slots=True)
class ForecastDatasetSourceManifest:
	dataset: DatasetManifest | None = None
	dataset_path: str | None = None

	def __post_init__(self) -> None:
		if not self.dataset_path and self.dataset is None:
			raise ValueError("Either dataset_path or dataset must be provided")


@dataclass(slots=True)
class ForecastLocalModelManifest:
	model_dir: str | None = None

	def is_configured(self) -> bool:
		return bool(self.model_dir and self.model_dir.strip())


@dataclass(slots=True)
class ForecastRegistryModelManifest:
	selection: RegistrySelectionManifest = field(
		default_factory=RegistrySelectionManifest
	)

	def is_configured(self) -> bool:
		return self.selection.is_configured()


@dataclass(slots=True)
class ForecastModelManifest:
	local: ForecastLocalModelManifest = field(default_factory=ForecastLocalModelManifest)
	registry: ForecastRegistryModelManifest = field(default_factory=ForecastRegistryModelManifest)
	source_preference: str = "auto"

	def __post_init__(self) -> None:
		self.source_preference = _normalize_source_preference(self.source_preference)
		if not self.local.is_configured() and not self.registry.is_configured():
			raise ValueError("At least one forecast model source must be configured: local or registry")

	def resolve_source_kind(self, execution_mode: str | None = None) -> str:
		has_local = self.local.is_configured()
		has_registry = self.registry.is_configured()
		if has_local and has_registry:
			if self.source_preference in {"local", "registry"}:
				return self.source_preference
			return "registry" if _resolve_execution_mode(execution_mode) == "airflow" else "local"
		if has_registry:
			return "registry"
		if has_local:
			return "local"
		raise ValueError("No forecast model source configured")


@dataclass(slots=True)
class ForecastForecastManifest:
	horizon_weeks: int | None = None

	def __post_init__(self) -> None:
		if self.horizon_weeks is not None and not 1 <= self.horizon_weeks <= 8:
			raise ValueError("Forecast horizon must stay within 1..8 weeks")


@dataclass(slots=True)
class ForecastPipelineManifest:
	model: ForecastModelManifest
	dataset: DatasetManifest | None = None
	dataset_path: str | None = None
	forecast: ForecastForecastManifest = field(default_factory=ForecastForecastManifest)
	runtime: RuntimeManifest = field(default_factory=RuntimeManifest)

	def __post_init__(self) -> None:
		ForecastDatasetSourceManifest(
			dataset=self.dataset,
			dataset_path=self.dataset_path,
		)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)

	@staticmethod
	def from_dict(data: dict[str, Any]) -> "ForecastPipelineManifest":
		input_payload = dict(data.get("input", {}))
		dataset_payload = data.get("dataset", input_payload.get("dataset"))
		dataset_path = data.get("dataset_path", input_payload.get("dataset_path"))
		model_payload = get_required_mapping_section(data, "model")
		registry_payload = dict(model_payload.get("registry", {}))
		selection_payload = dict(registry_payload.get("selection", {}))
		selection_alias = selection_payload.get("alias", selection_payload.get("registry_alias", "dev"))
		return ForecastPipelineManifest(
			dataset=DatasetManifest(**dataset_payload) if isinstance(dataset_payload, dict) else None,
			dataset_path=dataset_path,
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
			forecast=ForecastForecastManifest(**get_required_mapping_section(data, "forecast")),
			runtime=RuntimeManifest(**get_required_mapping_section(data, "runtime")),
		)

	@staticmethod
	def load(source: str | Path | dict[str, Any]) -> "ForecastPipelineManifest":
		if isinstance(source, dict):
			return ForecastPipelineManifest.from_dict(source)

		return ForecastPipelineManifest.from_dict(read_yaml_mapping(source))
