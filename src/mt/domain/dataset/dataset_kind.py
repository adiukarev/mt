from enum import StrEnum

from mt.infra.helper.enum import normalize_enum_by_key


class DatasetKind(StrEnum):
	M5 = "m5"
	FAVORITA = "favorita"
	SYNTHETIC = "synthetic"


ALLOWED_DATASET_KINDS = frozenset(DatasetKind)
DATASET_KIND_BY_VALUE = {dataset_kind.value: dataset_kind for dataset_kind in DatasetKind}


def normalize_dataset_kind(value: str | DatasetKind) -> DatasetKind:
	return normalize_enum_by_key(
		value,
		enum_type=DatasetKind,
		by_value=DATASET_KIND_BY_VALUE,
	)
