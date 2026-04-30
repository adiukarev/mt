from enum import StrEnum

from mt.infra.helper.enum import normalize_enum_by_key


class ModelName(StrEnum):
	NAIVE = "naive"
	SEASONAL_NAIVE = "seasonal_naive"
	ETS = "ets"
	RIDGE = "ridge"
	LIGHTGBM = "lightgbm"
	CATBOOST = "catboost"
	MLP = "mlp"
	NBEATS = "nbeats"


ALLOWED_MODEL_NAMES = frozenset(ModelName)
MODEL_NAME_BY_VALUE = {model_name.value: model_name for model_name in ModelName}


def normalize_model_name(value: str | ModelName) -> ModelName:
	return normalize_enum_by_key(
		value,
		enum_type=ModelName,
		by_value=MODEL_NAME_BY_VALUE,
	)
