from dataclasses import dataclass

from mt.domain.dataset.dataset_kind import ALLOWED_DATASET_KINDS, DatasetKind, normalize_dataset_kind

ALLOWED_AGGREGATION_LEVELS = {"category", "sku"}


@dataclass(slots=True)
class DatasetManifest:
	kind: DatasetKind = DatasetKind.M5
	path: str = "data/m5-forecasting-accuracy"
	aggregation_level: str = "category"
	target_name: str = "sales_units"
	week_anchor: str = "W-MON"
	series_limit: int | None = None
	series_allowlist: list[str] | None = None

	def __post_init__(self) -> None:
		self.kind = normalize_dataset_kind(self.kind)
		if self.kind not in ALLOWED_DATASET_KINDS:
			raise ValueError()
		if self.aggregation_level not in ALLOWED_AGGREGATION_LEVELS:
			raise ValueError()
		if self.week_anchor != "W-MON":
			raise ValueError()
		if self.series_limit is not None and self.series_limit <= 0:
			raise ValueError()
		if self.series_allowlist is not None:
			normalized = [str(item).strip() for item in self.series_allowlist if str(item).strip()]
			if not normalized:
				raise ValueError()
			self.series_allowlist = list(dict.fromkeys(normalized))
