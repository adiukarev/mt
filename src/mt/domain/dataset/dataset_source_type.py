from enum import StrEnum

from mt.infra.helper.enum import normalize_enum_by_key


class DatasetSourceType(StrEnum):
	LOCAL = "local"
	SYNTHETIC = "synthetic"
	POSTGRES = "postgres"
	CLICKHOUSE = "clickhouse"
	API = "api"


ALLOWED_DATASET_SOURCE_TYPES = frozenset(DatasetSourceType)
DATASET_SOURCE_TYPE_BY_VALUE = {
	source_type.value: source_type for source_type in DatasetSourceType
}
DATASET_SOURCE_TYPE_BY_VALUE["synthetic_refresh"] = DatasetSourceType.SYNTHETIC


def normalize_dataset_source_type(value: str | DatasetSourceType) -> DatasetSourceType:
	return normalize_enum_by_key(
		value,
		enum_type=DatasetSourceType,
		by_value=DATASET_SOURCE_TYPE_BY_VALUE,
	)
