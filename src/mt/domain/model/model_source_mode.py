from enum import StrEnum

from mt.infra.helper.enum import normalize_enum_by_key


class ModelSourceMode(StrEnum):
	LOCAL_ARTIFACTS = "local_artifacts"
	MLFLOW_RUN = "mlflow_run"
	MLFLOW_REGISTRY = "mlflow_registry"


ALLOWED_MODEL_SOURCE_MODES = frozenset(ModelSourceMode)
MODEL_SOURCE_MODE_BY_VALUE = {mode.value: mode for mode in ModelSourceMode}
MLFLOW_MODEL_SOURCE_MODES = frozenset(
	{ModelSourceMode.MLFLOW_RUN, ModelSourceMode.MLFLOW_REGISTRY}
)


def normalize_model_source_mode(value: str | ModelSourceMode) -> ModelSourceMode:
	return normalize_enum_by_key(
		value,
		enum_type=ModelSourceMode,
		by_value=MODEL_SOURCE_MODE_BY_VALUE,
	)
