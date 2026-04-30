from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(slots=True)
class ModelConfigRidgeManifest:
	alpha: float = 1.0

	def __post_init__(self) -> None:
		if self.alpha <= 0.0:
			raise ValueError()

	@classmethod
	def from_mapping(cls, payload: dict[str, Any] | None) -> "ModelConfigRidgeManifest":
		if payload is None:
			return cls()
		allowed_fields = {field.name for field in fields(cls)}
		return cls(**{key: value for key, value in payload.items() if key in allowed_fields})

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class ModelConfigLightGBMManifest:
	n_estimators: int = 300
	learning_rate: float = 0.05
	max_depth: int = 6
	num_leaves: int = 31
	min_child_samples: int = 20
	subsample: float = 0.9
	colsample_bytree: float = 0.9
	objective: str = "regression"
	objective_alpha: float = 0.9
	tweedie_variance_power: float = 1.5

	def __post_init__(self) -> None:
		if self.n_estimators < 1:
			raise ValueError()
		if self.learning_rate <= 0.0:
			raise ValueError()
		if self.max_depth < 1:
			raise ValueError()
		if self.num_leaves < 2 or self.min_child_samples < 1:
			raise ValueError()
		if not 0.0 < self.subsample <= 1.0 or not 0.0 < self.colsample_bytree <= 1.0:
			raise ValueError()
		if self.objective not in {"regression", "regression_l1", "quantile", "poisson", "tweedie"}:
			raise ValueError()
		if not 0.0 < self.objective_alpha < 1.0:
			raise ValueError()
		if not 1.0 <= self.tweedie_variance_power < 2.0:
			raise ValueError()

	@classmethod
	def from_mapping(cls, payload: dict[str, Any] | None) -> "ModelConfigLightGBMManifest":
		if payload is None:
			return cls()
		allowed_fields = {field.name for field in fields(cls)}
		return cls(**{key: value for key, value in payload.items() if key in allowed_fields})

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class ModelConfigCatBoostManifest:
	depth: int = 6
	iterations: int = 300
	learning_rate: float = 0.05
	l2_leaf_reg: float = 3.0
	loss_function: str = "RMSE"

	def __post_init__(self) -> None:
		if self.depth < 1 or self.iterations < 1:
			raise ValueError()
		if self.learning_rate <= 0.0 or self.l2_leaf_reg < 0.0:
			raise ValueError()
		if not self.loss_function:
			raise ValueError()

	@classmethod
	def from_mapping(cls, payload: dict[str, Any] | None) -> "ModelConfigCatBoostManifest":
		if payload is None:
			return cls()
		allowed_fields = {field.name for field in fields(cls)}
		return cls(**{key: value for key, value in payload.items() if key in allowed_fields})

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
