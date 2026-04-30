from enum import StrEnum

from mt.domain.model.model_name import ModelName


class ModelFamily(StrEnum):
	BASELINE = "baseline"
	STATISTICAL = "statistical"
	ML = "ml"
	DL = "dl"
	OTHER = "other"


MODEL_FAMILY_BY_NAME = {
	ModelName.NAIVE: ModelFamily.BASELINE,
	ModelName.SEASONAL_NAIVE: ModelFamily.BASELINE,
	ModelName.ETS: ModelFamily.STATISTICAL,
	ModelName.RIDGE: ModelFamily.ML,
	ModelName.LIGHTGBM: ModelFamily.ML,
	ModelName.CATBOOST: ModelFamily.ML,
	ModelName.MLP: ModelFamily.DL,
	ModelName.NBEATS: ModelFamily.DL,
}

BASELINE_MODEL_NAMES = frozenset({ModelName.NAIVE, ModelName.SEASONAL_NAIVE})
TREE_MODEL_NAMES = frozenset({ModelName.LIGHTGBM, ModelName.CATBOOST})
FEATURELESS_MODEL_NAMES = frozenset({ModelName.ETS})
HISTORY_WINDOW_MODEL_NAMES = frozenset({ModelName.NBEATS})
