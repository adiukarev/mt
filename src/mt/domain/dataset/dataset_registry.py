from dataclasses import dataclass
from pathlib import Path

from mt.domain.dataset.dataset_kind import DatasetKind, normalize_dataset_kind
from mt.infra.helper.enum import resolve_required_mapping

DATA_ROOT = Path("data")


@dataclass(frozen=True, slots=True)
class DatasetRegistryEntry:
	kind: DatasetKind
	extract_dir: Path
	archive_path: Path
	required_files: tuple[str, ...]
	download_url: str | None = None


DATASET_REGISTRY: dict[DatasetKind, DatasetRegistryEntry] = {
	DatasetKind.M5: DatasetRegistryEntry(
		kind=DatasetKind.M5,
		extract_dir=DATA_ROOT / "m5-forecasting-accuracy",
		archive_path=DATA_ROOT / "m5-forecasting-accuracy.zip",
		required_files=(
			"calendar.csv",
			"sales_train_evaluation.csv",
		),
		download_url="https://drive.google.com/file/d/1a27ZDdiWfhJwI00V_nlGVxwsn6dro4GP/view?usp=drive_link",
	),
	DatasetKind.FAVORITA: DatasetRegistryEntry(
		kind=DatasetKind.FAVORITA,
		extract_dir=DATA_ROOT / "favorita-grocery-sales-forecasting",
		archive_path=DATA_ROOT / "favorita-grocery-sales-forecasting.zip",
		required_files=(
			"train.csv",
			"items.csv",
			"stores.csv",
			"test.csv",
		),
		download_url="https://drive.google.com/file/d/1YYyLCHvwnpwTLgnmb-eDMyM1s6JTudx_/view?usp=drive_link",
	),
	DatasetKind.SYNTHETIC: DatasetRegistryEntry(
		kind=DatasetKind.SYNTHETIC,
		extract_dir=DATA_ROOT / "synthetic",
		archive_path=DATA_ROOT / "synthetic.zip",
		required_files=("dataset.csv",),
	),
}

ALLOWED_DATASET_REGISTRY_KINDS = frozenset(DATASET_REGISTRY)


def resolve_dataset_registry_entry(kind: str | DatasetKind) -> DatasetRegistryEntry:
	return resolve_required_mapping(
		kind,
		mapping=DATASET_REGISTRY,
		key_normalizer=normalize_dataset_kind,
	)
