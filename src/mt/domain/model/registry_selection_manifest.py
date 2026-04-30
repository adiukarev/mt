from dataclasses import dataclass

from mt.domain.model.manifest_support import (
	normalize_optional_string,
	normalize_required_string,
)


@dataclass(slots=True)
class RegistrySelectionManifest:
	dataset_kind: str | None = None
	aggregation_level: str | None = None
	target_name: str | None = None
	dag_id: str | None = None
	alias: str = "dev"
	metric_name: str = "WAPE"
	higher_is_better: bool = False

	def __post_init__(self) -> None:
		self.dataset_kind = normalize_optional_string(self.dataset_kind)
		self.aggregation_level = normalize_optional_string(self.aggregation_level)
		self.target_name = normalize_optional_string(self.target_name)
		self.dag_id = normalize_optional_string(self.dag_id)
		self.alias = normalize_required_string(
			self.alias,
			"alias",
		)
		self.metric_name = normalize_required_string(self.metric_name, "metric_name")

	def is_configured(self) -> bool:
		has_dataset_scope = all(
			(
				self.dataset_kind,
				self.aggregation_level,
				self.target_name,
			)
		)
		return has_dataset_scope or self.dag_id is not None

	@property
	def registry_alias(self) -> str:
		return self.alias
