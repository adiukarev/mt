from enum import StrEnum

from mt.domain.dataset.dataset_kind import DatasetKind, normalize_dataset_kind
from mt.infra.helper.enum import normalize_enum_by_key, resolve_required_mapping


class DatasetAdapterName(StrEnum):
	M5 = "m5"
	FAVORITA = "favorita"
	SYNTHETIC = "synthetic"


ALLOWED_DATASET_ADAPTER_NAMES = frozenset(DatasetAdapterName)
DATASET_ADAPTER_NAME_BY_VALUE = {
	dataset_adapter_name.value: dataset_adapter_name for dataset_adapter_name in DatasetAdapterName
}

DATASET_ADAPTER_BY_KIND: dict[DatasetKind, DatasetAdapterName] = {
	DatasetKind.M5: DatasetAdapterName.M5,
	DatasetKind.FAVORITA: DatasetAdapterName.FAVORITA,
	DatasetKind.SYNTHETIC: DatasetAdapterName.SYNTHETIC,
}


def normalize_dataset_adapter_name(value: str | DatasetAdapterName) -> DatasetAdapterName:
	return normalize_enum_by_key(
		value,
		enum_type=DatasetAdapterName,
		by_value=DATASET_ADAPTER_NAME_BY_VALUE,
	)


def resolve_dataset_adapter_name(kind: str | DatasetKind) -> DatasetAdapterName:
	return resolve_required_mapping(
		kind,
		mapping=DATASET_ADAPTER_BY_KIND,
		key_normalizer=normalize_dataset_kind,
	)
